import hashlib
import sys
import threading
import socket
import time
import select
import errno
from typing import Union


class TcpProxy(object):
    __header = 'PROXY#'
    __close_header = 'CLOSE#'
    __uid_length = 64
    __socket_test_timeout = 3
    __max_sockets = 100
    __wait_timeout = 300  # make sure we do not collect lost sockets, and drop it after 5 minutes
    __default_packet_size = 4096

    def __init__(self,
                 listen_port=8868, target_port=8878, proxy_state=None, verbose=None,
                 keep_connection=False, is_connection_server=False):
        # type: (int, int, dict, bool, bool, bool) -> ()
        self.listen_ip = '127.0.0.1'
        self.target_ip = '127.0.0.1'
        self.logfile = None  # sys.stdout
        self.listen_port = listen_port
        self.target_port = target_port
        self.proxy_state = proxy_state or {}
        self.verbose = verbose
        self.proxy_socket = None
        self.active_local_sockets = {}
        self.close_local_sockets = set()
        self.keep_connection = keep_connection
        self.keep_connection_server = keep_connection and is_connection_server
        self.keep_connection_client = keep_connection and not is_connection_server
        # set max number of open files
        # noinspection PyBroadException
        try:
            if sys.platform == 'win32':
                import ctypes
                ctypes.windll.msvcrt._setmaxstdio(max(2048, ctypes.windll.msvcrt._getmaxstdio()))  # noqa
            else:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (max(4096, soft), hard))
        except Exception:
            pass
        self._proxy_daemon_thread = threading.Thread(target=self.daemon)
        self._proxy_daemon_thread.setDaemon(True)
        self._proxy_daemon_thread.start()

    def get_thread(self):
        return self._proxy_daemon_thread

    @staticmethod
    def receive_from(s, size=0):
        # type: (socket.socket, int) -> bytes
        # receive data from a socket until no more data is there
        b = b""
        while True:
            data = s.recv(size-len(b) if size else TcpProxy.__default_packet_size)
            b += data
            if size and len(b) < size:
                continue
            if size or not data or len(data) < TcpProxy.__default_packet_size:
                break
        return b

    @staticmethod
    def send_to(s, data):
        # type: (socket.socket, Union[str, bytes]) -> ()
        s.send(data.encode() if isinstance(data, str) else data)

    def start_proxy_thread(self, local_socket, uuid, init_data):
        try:
            remote_socket = self._open_remote_socket(local_socket)
        except Exception as ex:
            self.vprint('Exception {}: {}'.format(type(ex), ex))
            return
        while True:
            try:
                init_data_ = init_data
                init_data = None
                self._process_socket_proxy(local_socket, remote_socket, uuid=uuid, init_data=init_data_)
                return
            except Exception as ex:
                self.vprint('Exception {}: {}'.format(type(ex), ex))
                time.sleep(0.1)

    def _open_remote_socket(self, local_socket):
        # This method is executed in a thread. It will relay data between the local
        # host and the remote host, while letting modules work on the data before
        # passing it on.
        remote_socket = None
        while True:
            if remote_socket:
                remote_socket.close()
            remote_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            timeout = 60
            try:
                remote_socket.settimeout(timeout)
                remote_socket.connect((self.target_ip, self.target_port))
                msg = 'Connected to {}'.format(remote_socket.getpeername())
                self.vprint(msg)
                self.log(msg)
            except socket.error as serr:
                if serr.errno == errno.ECONNREFUSED:
                    # for s in [remote_socket, local_socket]:
                    #     s.close()
                    msg = '{}, {}:{} - Connection refused'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), self.target_ip, self.target_port)
                    self.vprint(msg)
                    self.log(msg)
                    # return None
                    self.proxy_state['reconnect'] = True
                    time.sleep(1)
                    continue
                elif serr.errno == errno.ETIMEDOUT:
                    # for s in [remote_socket, local_socket]:
                    #     s.close()
                    msg = '{}, {}:{} - Connection connection timed out'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), self.target_ip, self.target_port)
                    self.vprint(msg)
                    self.log(msg)
                    # return None
                    self.proxy_state['reconnect'] = True
                    time.sleep(1)
                    continue
                else:
                    self.vprint("Connection error {}".format(serr.errno))
                    for s in [remote_socket, local_socket]:
                        s.close()
                    raise serr
            break

        return remote_socket

    def _process_socket_proxy(self, local_socket, remote_socket, uuid=None, init_data=None):
        # This method is executed in a thread. It will relay data between the local
        # host and the remote host, while letting modules work on the data before
        # passing it on.
        timeout = 60

        # if we are self.keep_connection_client we need to generate uuid, send it
        if self.keep_connection_client:
            if uuid is None:
                uuid = hashlib.sha256('{}{}'.format(time.time(), local_socket.getpeername()).encode()).hexdigest()
            self.vprint('sending UUID {}'.format(uuid))
            self.send_to(remote_socket, self.__header + uuid)

        # check if we need to send init_data
        if init_data:
            self.vprint('sending init data {}'.format(len(init_data)))
            self.send_to(remote_socket, init_data)

        # This loop ends when no more data is received on either the local or the
        # remote socket
        running = True
        while running:
            read_sockets, _, _ = select.select([remote_socket, local_socket], [], [])

            for sock in read_sockets:
                try:
                    peer = sock.getpeername()
                except socket.error as serr:
                    if serr.errno == errno.ENOTCONN:
                        # kind of a blind shot at fixing issue #15
                        # I don't yet understand how this error can happen,
                        # but if it happens I'll just shut down the thread
                        # the connection is not in a useful state anymore
                        for s in [remote_socket, local_socket]:
                            s.close()
                        running = False
                        break
                    else:
                        self.vprint("{}: Socket exception in start_proxy_thread".format(
                            time.strftime('%Y-%m-%d %H:%M:%S')))
                        raise serr

                data = self.receive_from(sock)
                self.log('Received %d bytes' % len(data))

                if sock == local_socket:
                    if len(data):
                        # log(args.logfile, b'< < < out\n' + data)
                        self.send_to(remote_socket, data)
                    else:
                        msg = "Connection from local client %s:%d closed" % peer
                        self.vprint(msg)
                        self.log(msg)
                        local_socket.close()
                        if not self.keep_connection or not uuid:
                            remote_socket.close()
                            running = False
                        elif self.keep_connection_server:
                            # test remote socket
                            self.vprint('waiting for reconnection, sleep 1 sec')
                            tic = time.time()
                            while uuid not in self.close_local_sockets and \
                                    self.active_local_sockets.get(uuid, {}).get('local_socket') == local_socket:
                                time.sleep(1)
                                self.vprint('wait local reconnect [{}]'.format(uuid))
                                if time.time() - tic > self.__wait_timeout:
                                    remote_socket.close()
                                    running = False
                                    break
                            if not running:
                                break

                            self.vprint('done waiting')
                            if uuid in self.close_local_sockets:
                                self.vprint('client closed connection')
                                remote_socket.close()
                                running = False
                                self.close_local_sockets.remove(uuid)
                            else:
                                self.vprint('reconnecting local client')
                                local_socket = self.active_local_sockets.get(uuid, {}).get('local_socket')

                        elif self.keep_connection_client:
                            # send UUID goodbye message
                            self.vprint('client {} closing socket'.format(uuid))
                            remote_socket.close()
                            running = False

                        break

                elif sock == remote_socket:
                    if len(data):
                        # log(args.logfile, b'> > > in\n' + data)
                        self.send_to(local_socket, data)
                    else:
                        msg = "Connection to remote server %s:%d closed" % peer
                        self.vprint(msg)
                        self.log(msg)
                        remote_socket.close()
                        if self.keep_connection_client and uuid:
                            # self.proxy_state['reconnect'] = True
                            self.vprint('Wait for remote reconnect')
                            time.sleep(1)
                            return self.start_proxy_thread(local_socket, uuid=uuid, init_data=None)
                        else:
                            local_socket.close()
                            running = False
                            break

        # remove the socket from the global list
        if uuid:
            self.active_local_sockets.pop(uuid, None)
            if self.keep_connection_client:
                self._send_remote_close_msg(timeout, uuid)

    def _send_remote_close_msg(self, timeout, uuid):
        if not self.keep_connection_client or not uuid:
            return
        try:
            self.vprint('create new control socket')
            control_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            control_socket.settimeout(timeout)
            control_socket.connect((self.target_ip, self.target_port))
            self.vprint('send close header [{}]'.format(uuid))
            self.send_to(control_socket, self.__close_header + uuid)
            self.vprint('close control_socket')
            control_socket.close()
        except Exception as ex:
            self.vprint('Error sending close header, '.format(ex))

    def log(self, message, message_only=False):
        # if message_only is True, only the message will be logged
        # otherwise the message will be prefixed with a timestamp and a line is
        # written after the message to make the log file easier to read
        handle = self.logfile
        if handle is None:
            return
        if not isinstance(message, bytes):
            message = bytes(message, 'ascii')
        if not message_only:
            logentry = bytes("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), str(time.time())), 'ascii')
        else:
            logentry = b''
        logentry += message
        if not message_only:
            logentry += b'\n' + b'-' * 20 + b'\n'
        handle.write(logentry.decode())

    def vprint(self, msg):
        # this will print msg, but only if is_verbose is True
        if self.verbose:
            print(msg)

    def daemon(self):
        # this is the socket we will listen on for incoming connections
        self.proxy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.proxy_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.proxy_socket.bind((self.listen_ip, self.listen_port))
        except socket.error as e:
            print(e.strerror)
            sys.exit(5)

        self.proxy_socket.listen(self.__max_sockets)
        # endless loop
        while True:
            try:
                in_socket, in_addrinfo = self.proxy_socket.accept()
                msg = 'Connection from %s:%d' % in_addrinfo  # noqa
                self.vprint(msg)
                self.log(msg)
                uuid = None
                init_data = None
                if self.keep_connection_server:
                    read_sockets, _, _ = select.select([in_socket], [], [])
                    if read_sockets:
                        data = self.receive_from(in_socket, size=self.__uid_length + len(self.__header))
                        self.vprint('Reading header [{}]'.format(len(data)))
                        if len(data) == self.__uid_length + len(self.__header):
                            # noinspection PyBroadException
                            try:
                                header = data.decode()
                            except Exception:
                                header = None
                            if header.startswith(self.__header):
                                uuid = header[len(self.__header):]
                                self.vprint('Reading UUID [{}] {}'.format(len(data), uuid))
                            elif header.startswith(self.__close_header):
                                uuid = header[len(self.__close_header):]
                                self.vprint('Closing UUID [{}] {}'.format(len(data), uuid))
                                self.close_local_sockets.add(uuid)
                                continue
                            else:
                                init_data = data
                        else:
                            init_data = data

                    if self.active_local_sockets and uuid is not None:
                        self.vprint('Check waiting threads')
                        # noinspection PyBroadException
                        try:
                            if uuid in self.active_local_sockets:
                                self.vprint('Updating thread uuid {}'.format(uuid))
                                self.active_local_sockets[uuid]['local_socket'] = in_socket
                                continue
                        except Exception:
                            pass

                if uuid:
                    self.active_local_sockets[uuid] = {'local_socket': in_socket}

                # check if thread is waiting
                proxy_thread = threading.Thread(target=self.start_proxy_thread, args=(in_socket, uuid, init_data))
                proxy_thread.setDaemon(True)
                self.log("Starting proxy thread " + proxy_thread.name)
                proxy_thread.start()
            except Exception as ex:
                msg = 'Exception: {}'.format(ex)
                self.vprint(msg)
                self.log(msg)
