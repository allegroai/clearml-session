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
            # noinspection PyBroadException
            try:
                self.forward.connect((host, port))
                return self.forward
            except Exception:
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
        while 1:
            time.sleep(self.delay)
            # noinspection PyBroadException
            try:
                inputready, outputready, exceptready = select.select(self.input_list, [], [])
            except Exception:
                continue
            for s in inputready:
                if s == self.server:
                    # noinspection PyBroadException
                    try:
                        self.on_accept()
                    except Exception:
                        pass
                    break

                # noinspection PyBroadException
                try:
                    data = s.recv(self.buffer_size)
                except ConnectionResetError:
                    # this will trigger on_close
                    data = []
                except Exception:
                    continue

                if len(data) == 0:
                    # noinspection PyBroadException
                    try:
                        self.on_close(s)
                    except Exception:
                        pass
                    break
                else:
                    # noinspection PyBroadException
                    try:
                        self.on_recv(s, data)
                    except Exception:
                        pass

    def on_accept(self):
        clientsock, clientaddr = self.server.accept()
        forward = None
        for i in range(self.max_timeout_for_remote_connection):
            forward = self.Forward().start(self.tgthost, self.tgtport)
            if forward:
                break
            # print('waiting for remote...')
            time.sleep(1)

        if forward:
            # print("{0} has connected".format(clientaddr))
            self.input_list.append(clientsock)
            self.input_list.append(forward)
            self.channel[clientsock] = forward
            self.channel[forward] = clientsock
            sidbase = "{0}_{1}_{2}_{3}".format(self.tgthost, self.tgtport, clientaddr[0], clientaddr[1])
            self.sidmap[clientsock] = (sidbase, 1)
            self.sidmap[forward] = (sidbase, -1)
        else:
            # print("Can't establish connection with remote server.\n"
            #       "Closing connection with client side{0}".format(clientaddr))
            clientsock.close()

    def on_close(self, s):
        # logger.info("{0} has disconnected".format(self.s.getpeername()))
        # print("has disconnected")

        self.input_list.remove(s)
        self.input_list.remove(self.channel[s])
        out = self.channel[s]
        self.channel[out].close()
        self.channel[s].close()
        del self.channel[out]
        del self.channel[s]
        del self.sidmap[out]
        del self.sidmap[s]

    def on_recv(self, s, data):
        _sidbase = self.sidmap[s][0]
        _c_or_s = self.sidmap[s][1]
        # logger.debug(ctrl_less(data.strip()))
        self.channel[s].send(data)
