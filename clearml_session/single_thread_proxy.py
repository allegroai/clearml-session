import threading
import socket
import time
import select
import sys


class SingleThreadProxy(object):
    max_timeout_for_remote_connection = 60

    class Forward(object):
        def __init__(self):
            self.forward = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        def start(self, host, port):
            try:
                self.forward.connect((host, port))
                return self.forward
            except Exception as e:
                return False

    def __init__(self, port, tgtport, host="127.0.0.1", tgthost="127.0.0.1",
                 buffer_size=4096, delay=0.0001, state=None):
        self.input_list = []
        self.channel = {}
        self.sidmap = {}
        self.state = state or {}

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

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(100)
        self.tgthost, self.tgtport = tgthost, tgtport
        self.buffer_size, self.delay = buffer_size, delay
        self._proxy_daemon_thread = threading.Thread(target=self.main_loop)
        self._proxy_daemon_thread.setDaemon(True)
        self._proxy_daemon_thread.start()

    def main_loop(self):
        self.input_list.append(self.server)
        ss = select.select
        while 1:
            time.sleep(self.delay)
            try:
                inputready, outputready, exceptready = ss(self.input_list, [], [])
            except:
                continue
            for self.s in inputready:
                if self.s == self.server:
                    try:
                        self.on_accept()
                    except:
                        pass
                    break

                try:
                    self.data = self.s.recv(self.buffer_size)
                except:
                    continue
                if len(self.data) == 0:
                    try:
                        self.on_close()
                    except:
                        pass
                    break
                else:
                    try:
                        self.on_recv()
                    except:
                        pass

    def on_accept(self):
        clientsock, clientaddr = self.server.accept()
        for i in range(self.max_timeout_for_remote_connection):
            forward = self.Forward().start(self.tgthost, self.tgtport)
            if forward:
                break
            # print('waiting for remote...')
            time.sleep(1)

        if forward:
            # logger.info("{0} has connected".format(clientaddr))
            self.input_list.append(clientsock)
            self.input_list.append(forward)
            self.channel[clientsock] = forward
            self.channel[forward] = clientsock
            _sidbase = "{0}_{1}_{2}_{3}".format(self.tgthost, self.tgtport, clientaddr[0], clientaddr[1])
            self.sidmap[clientsock] = (_sidbase, 1)
            self.sidmap[forward] = (_sidbase, -1)
        else:
            # logger.warn("Can't establish connection with remote server.\n"
            #             "Closing connection with client side{0}".format(clientaddr))
            clientsock.close()

    def on_close(self):
        # logger.info("{0} has disconnected".format(self.s.getpeername()))

        self.input_list.remove(self.s)
        self.input_list.remove(self.channel[self.s])
        out = self.channel[self.s]
        self.channel[out].close()
        self.channel[self.s].close()
        del self.channel[out]
        del self.channel[self.s]
        del self.sidmap[out]
        del self.sidmap[self.s]

    def on_recv(self):
        _sidbase = self.sidmap[self.s][0]
        _c_or_s = self.sidmap[self.s][1]
        data = self.data
        # logger.debug(ctrl_less(data.strip()))
        self.channel[self.s].send(data)
