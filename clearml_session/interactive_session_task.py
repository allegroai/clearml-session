import base64
import json
import os
import socket
import subprocess
import sys
from copy import deepcopy
import getpass
from tempfile import mkstemp, gettempdir
from time import sleep

import psutil
import requests
from clearml import Task, StorageManager
from clearml.backend_api import Session
from pathlib2 import Path
from tcp_proxy import TcpProxy

# noinspection SpellCheckingInspection
default_ssh_fingerprint = {
    'ssh_host_ecdsa_key':
        r"-----BEGIN EC PRIVATE KEY-----"+"\n"
        r"MHcCAQEEIOCAf3KEN9Hrde53rqQM4eR8VfCnO0oc4XTEBw0w6lCfoAoGCCqGSM49"+"\n"
        r"AwEHoUQDQgAEn/LlC/1UN1q6myfjs03LJdHY2LB0b1hBjAsLvQnDMt8QE6Rml3UF"+"\n"
        r"QK/UFw4mEqCFCD+dcbyWqFsKxTm6WtFStg=="+"\n"
        r"-----END EC PRIVATE KEY-----"+"\n",

    'ssh_host_ed25519_key':
        r"-----BEGIN OPENSSH PRIVATE KEY-----"+"\n"
        r"b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"+"\n"
        r"QyNTUxOQAAACDvweeJHnUKtwY7/WRqDJEZTDk8AajWKFt/BXmEI3+A8gAAAJiEMTXOhDE1"+"\n"
        r"zgAAAAtzc2gtZWQyNTUxOQAAACDvweeJHnUKtwY7/WRqDJEZTDk8AajWKFt/BXmEI3+A8g"+"\n"
        r"AAAEBCHpidTBUN3+W8s3qRNkyaJpA/So4vEqDvOhseSqJeH+/B54kedQq3Bjv9ZGoMkRlM"+"\n"
        r"OTwBqNYoW38FeYQjf4DyAAAAEXJvb3RAODQ1NmQ5YTdlYTk4AQIDBA=="+"\n"
        r"-----END OPENSSH PRIVATE KEY-----"+"\n",

    'ssh_host_rsa_key':
        r"-----BEGIN RSA PRIVATE KEY-----"+"\n"
        r"MIIEowIBAAKCAQEAs8R3BrinMM/k9Jak7UqsoONqLQoasYgkeBVOOfRJ6ORYWW5R"+"\n"
        r"WLkYnPPUGRpbcoM1Imh7ODBgKzs0mh5/j3y0SKP/MpvT4bf38e+QGjuC+6fR4Ah0"+"\n"
        r"L5ohGIMyqhAiBoXgj0k2BE6en/4Rb3BwNPMocCTus82SwajzMNgWneRC6GCq2M0n"+"\n"
        r"0PWenhS0IQz7jUlw3JU8z6T3ROPiMBPU7ubBhiNlAzMYPr76Z7J6ZNrCclAvdGkI"+"\n"
        r"YxK7RNq0HwfoUj0UFD9iaEHswDIlNc34p93lP6GIAbh7uVYfGhg4z7HdBoN2qweN"+"\n"
        r"szo7iQX9N8EFP4WfpLzNFteThzgN/bdso8iv0wIDAQABAoIBAQCPvbF64110b1dg"+"\n"
        r"p7AauVINl6oHd4PensCicE7LkmUi3qsyXz6WVfKzVVgr9mJWz0lGSQr14+CR0NZ/"+"\n"
        r"wZE393vkdZWSLv2eB88vWeH8x8c1WHw9yiS1B2YdRpLVXu8GDjh/+gdCLGc0ASCJ"+"\n"
        r"3fsqq5+TBEUF6oPFbEWAsdhryeAiFAokeIVEKkxRnIDvPCP6i0evUHAxEP+wOngu"+"\n"
        r"4XONkixNmATNa1jP2YAjmh3uQbAf2BvDZuywJmqV8fqZa/BwuK3W+R/92t0ySZ5Q"+"\n"
        r"Z7RCZzPzFvWY683/Cfx5+BH3XcIetbcZ/HKuc+TdBvvFgqrLNIJ4OXMp3osjZDMO"+"\n"
        r"YZIE6DdBAoGBAOG8cgm2N+Kl2dl0q1r4S+hf//zPaDorNasvcXJcj/ypy1MdmDXt"+"\n"
        r"whLSAuTN4r8axgbuws2Z870pIGd28koqg78U+pOPabkphloo8Fc97RO28ZJCK2g0"+"\n"
        r"/prPgwSYymkhrvwdzIbI11BPL/rr9cLJ1eYDnzGDSqvXJDL79XxrzwMzAoGBAMve"+"\n"
        r"ULkfqaYVlgY58d38XruyCpSmRSq39LTeTYRWkJTNFL6rkqL9A69z/ITdpSStEuR8"+"\n"
        r"8MXQSsPz8xUhFrA2bEjW7AT0r6OqGbjljKeh1whYOfgGfMKQltTfikkrf5w0UrLw"+"\n"
        r"NQ8USfpwWdFnBGQG0yE/AFknyLH14/pqfRlLzaDhAoGAcN3IJxL03l4OjqvHAbUk"+"\n"
        r"PwvA8qbBdlQkgXM3RfcCB1LeVrB1aoF2h/J5f+1xchvw54Z54FMZi3sEuLbAblTT"+"\n"
        r"irbyktUiB3K7uli90uEjqLfQEVEEYxYcN0uKNsIucmJlG6nKmZnSDlWJp+xS9RH1"+"\n"
        r"4QvujNMYgtMPRm60T4GYAAECgYB6J9LMqik4CDUls/C2J7MH2m22lk5Zg3JQMefW"+"\n"
        r"xRvK3XtxqFKr8NkVd3U2k6yRZlcsq6SFkwJJmdHsti/nFCUcHBO+AHOBqLnS7VCz"+"\n"
        r"XSkAqgTKFfEJkCOgl/U/VJ4ZFcz7xSy1xV1yf4GCFK0v1lsJz7tAsLLz1zdsZARj"+"\n"
        r"dOVYYQKBgC3IQHfd++r9kcL3+vU7bDVU4aKq0JFDA79DLhKDpSTVxqTwBT+/BIpS"+"\n"
        r"8z79zBTjNy5gMqxZp/SWBVWmsO8d7IUk9O2L/bMhHF0lOKbaHQQ9oveCzIwDewcf"+"\n"
        r"5I45LjjGPJS84IBYv4NElptRk/2eFFejr75xdm4lWfpLb1SXPOPB"+"\n"
        r"-----END RSA PRIVATE KEY-----"+"\n",

    'ssh_host_rsa_key__pub':
        r'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCzxHcGuKcwz+T0lqTtSqyg42otChqxiCR4FU459Eno5FhZblFYuRic89QZGlt'
        r'ygzUiaHs4MGArOzSaHn+PfLRIo/8ym9Pht/fx75AaO4L7p9HgCHQvmiEYgzKqECIGheCPSTYETp6f/hFvcHA08yhwJO6zzZLBqPM'
        r'w2Bad5ELoYKrYzSfQ9Z6eFLQhDPuNSXDclTzPpPdE4+IwE9Tu5sGGI2UDMxg+vvpnsnpk2sJyUC90aQhjErtE2rQfB+hSPRQUP2Jo'
        r'QezAMiU1zfin3eU/oYgBuHu5Vh8aGDjPsd0Gg3arB42zOjuJBf03wQU/hZ+kvM0W15OHOA39t2yjyK/T',
    'ssh_host_ecdsa_key__pub':
        r'ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBJ/y5Qv9VDdaupsn47NNyyXR2Niwd'
        r'G9YQYwLC70JwzLfEBOkZpd1BUCv1BcOJhKghQg/nXG8lqhbCsU5ulrRUrY=',
    'ssh_host_ed25519_key__pub': None,
}
config_section_name = 'interactive_session'
config_object_section_ssh = 'SSH'
config_object_section_bash_init = 'interactive_init_script'


__allocated_ports = []


def get_free_port(range_min, range_max):
    global __allocated_ports
    used_ports = [i.laddr.port for i in psutil.net_connections()]
    port = next(i for i in range(range_min, range_max) if i not in used_ports and i not in __allocated_ports)
    __allocated_ports.append(port)
    return port


def init_task(param, a_default_ssh_fingerprint):
    # initialize ClearML
    Task.add_requirements('jupyter')
    Task.add_requirements('jupyterlab')
    Task.add_requirements('jupyterlab_git')
    task = Task.init(
        project_name="DevOps", task_name="Allocate Jupyter Notebook Instance", task_type=Task.TaskTypes.service)

    # Add jupyter server base folder
    if Session.check_min_api_version('2.13'):
        param.pop('user_key', None)
        param.pop('user_secret', None)
        param.pop('ssh_password', None)
        task.connect(param, name=config_section_name)
        # noinspection PyProtectedMember
        runtime_prop = dict(task._get_runtime_properties())
        # remove the user key/secret the moment we have it
        param['user_key'] = runtime_prop.pop('_user_key', None)
        param['user_secret'] = runtime_prop.pop('_user_secret', None)
        # no need to reset, we will need it
        param['ssh_password'] = runtime_prop.get('_ssh_password')
        # Force removing properties
        # noinspection PyProtectedMember
        task._edit(runtime=runtime_prop)
        task.reload()
    else:
        task.connect(param, name=config_section_name)

    # connect ssh finger print configuration (with fallback if section is missing)
    old_default_ssh_fingerprint = deepcopy(a_default_ssh_fingerprint)
    try:
        task.connect_configuration(configuration=a_default_ssh_fingerprint, name=config_object_section_ssh)
    except (TypeError, ValueError):
        a_default_ssh_fingerprint.clear()
        a_default_ssh_fingerprint.update(old_default_ssh_fingerprint)
    if param.get('default_docker'):
        task.set_base_docker("{} --network host".format(param['default_docker']))
    # leave local process, only run remotely
    task.execute_remotely()
    return task


def setup_os_env(param):
    # get rid of all the runtime ClearML
    preserve = (
        "_API_HOST",
        "_WEB_HOST",
        "_FILES_HOST",
        "_CONFIG_FILE",
        "_API_ACCESS_KEY",
        "_API_SECRET_KEY",
        "_API_HOST_VERIFY_CERT",
        "_DOCKER_IMAGE",
        "_DOCKER_BASH_SCRIPT",
    )
    # set default docker image, with network configuration
    if param.get('default_docker', '').strip():
        os.environ["CLEARML_DOCKER_IMAGE"] = param['default_docker'].strip()

    # setup os environment
    env = deepcopy(os.environ)
    for key in os.environ:
        # only set CLEARML_ remove any TRAINS_
        if key.startswith("TRAINS") or (key.startswith("CLEARML") and not any(key.endswith(p) for p in preserve)):
            env.pop(key, None)

    return env


def monitor_jupyter_server(fd, local_filename, process, task, jupyter_port, hostnames):
    # todo: add auto spin down see: https://tljh.jupyter.org/en/latest/topic/idle-culler.html
    # print stdout/stderr
    prev_line_count = 0
    process_running = True
    token = None
    while process_running:
        process_running = False
        try:
            process.wait(timeout=2.0 if not token else 15.0)
        except subprocess.TimeoutExpired:
            process_running = True

        # noinspection PyBroadException
        try:
            with open(local_filename, "rt") as f:
                # read new lines
                new_lines = f.readlines()
                if not new_lines:
                    continue
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
        except Exception:
            continue

        print("".join(new_lines))
        prev_line_count += len(new_lines)
        # if we already have the token, do nothing, just monitor
        if token:
            continue

        # update task with jupyter notebook server links (port / token)
        line = ''
        for line in new_lines:
            if "http://" not in line and "https://" not in line:
                continue
            parts = line.split('?token=', 1)
            if len(parts) != 2:
                continue
            token = parts[1]
            port = parts[0].split(':')[-1]
            # try to cast to int
            try:
                port = int(port)  # noqa
            except (TypeError, ValueError):
                continue
            break
        # we could not locate the token, try again
        if not token:
            continue

        # we ignore the reported port, because jupyter server will get confused
        # if we have multiple servers running and will point to the wrong port/server
        task.set_parameter(name='properties/jupyter_port', value=str(jupyter_port))
        jupyter_url = '{}://{}:{}?token={}'.format(
            'https' if "https://" in line else 'http',
            hostnames, jupyter_port, token
        )

        # update the task with the correct links and token
        if Session.check_min_api_version("2.13"):
            # noinspection PyProtectedMember
            runtime_prop = task._get_runtime_properties()
            runtime_prop['_jupyter_token'] = str(token)
            runtime_prop['_jupyter_url'] = str(jupyter_url)
            # noinspection PyProtectedMember
            task._set_runtime_properties(runtime_prop)
        else:
            task.set_parameter(name='properties/jupyter_token', value=str(token))
            task.set_parameter(name='properties/jupyter_url', value=jupyter_url)

        print('\nJupyter Lab URL: {}\n'.format(jupyter_url))

    # cleanup
    # noinspection PyBroadException
    try:
        os.close(fd)
    except Exception:
        pass
    # noinspection PyBroadException
    try:
        os.unlink(local_filename)
    except Exception:
        pass


def start_vscode_server(hostname, hostnames, param, task, env):
    if not param.get("vscode_server"):
        return

    # get vscode version and python extension version
    # they are extremely flaky, this combination works, most do not.
    vscode_version = '3.12.0'
    python_ext_version = '2021.10.1365161279'
    if param.get("vscode_version"):
        vscode_version_parts = param.get("vscode_version").split(':')
        vscode_version = vscode_version_parts[0]
        if len(vscode_version_parts) > 1:
            python_ext_version = vscode_version_parts[1]

    # make a copy of env and remove the pythonpath from it.
    env = dict(**env)
    env.pop('PYTHONPATH', None)

    python_ext_download_link = \
        os.environ.get("CLEARML_SESSION_VSCODE_PY_EXT") or \
        'https://github.com/microsoft/vscode-python/releases/download/{}/ms-python-release.vsix'
    code_server_deb_download_link = \
        os.environ.get("CLEARML_SESSION_VSCODE_SERVER_DEB") or \
        'https://github.com/cdr/code-server/releases/download/v{version}/code-server_{version}_amd64.deb'

    pre_installed = False
    python_ext = None

    # find a free tcp port
    port = get_free_port(9000, 9100)

    if os.geteuid() == 0:
        # check if preinstalled
        # noinspection PyBroadException
        try:
            vscode_path = subprocess.check_output('which code-server', shell=True).decode().strip()
            pre_installed = bool(vscode_path)
        except Exception:
            vscode_path = None

        if not vscode_path:
            # installing VSCODE:
            try:
                python_ext = StorageManager.get_local_copy(
                    python_ext_download_link.format(python_ext_version),
                    extract_archive=False)
                code_server_deb = StorageManager.get_local_copy(
                    code_server_deb_download_link.format(version=vscode_version),
                    extract_archive=False)
                os.system("dpkg -i {}".format(code_server_deb))
            except Exception as ex:
                print("Failed installing vscode server: {}".format(ex))
                return
            vscode_path = 'code-server'
    else:
        python_ext = None
        pre_installed = True
        # check if code-server exists
        # noinspection PyBroadException
        try:
            vscode_path = subprocess.check_output('which code-server', shell=True).decode().strip()
            assert vscode_path
        except Exception:
            print('Error: Cannot install code-server (not root) and could not find code-server executable, skipping.')
            task.set_parameter(name='properties/vscode_port', value=str(-1))
            return

    cwd = (
        os.path.expandvars(os.path.expanduser(param["user_base_directory"]))
        if param["user_base_directory"]
        else os.getcwd()
    )
    # make sure we have the needed cwd
    # noinspection PyBroadException
    try:
        Path(cwd).mkdir(parents=True, exist_ok=True)
    except Exception:
        print("Warning: failed setting user base directory [{}] reverting to ~/".format(cwd))
        cwd = os.path.expanduser("~/")

    print("Running VSCode Server on {} [{}] port {} at {}".format(hostname, hostnames, port, cwd))
    print("VSCode Server available: http://{}:{}/\n".format(hostnames, port))
    user_folder = os.path.join(cwd, ".vscode/user/")
    exts_folder = os.path.join(cwd, ".vscode/exts/")

    try:
        fd, local_filename = mkstemp()
        if pre_installed:
            user_folder = os.path.expanduser("~/.local/share/code-server/")
            if not os.path.isdir(user_folder):
                user_folder = None
                exts_folder = None
            else:
                exts_folder = os.path.expanduser("~/.local/share/code-server/extensions/")
        else:
            subprocess.Popen(
                [
                    vscode_path,
                    "--auth",
                    "none",
                    "--bind-addr",
                    "127.0.0.1:{}".format(port),
                    "--user-data-dir", user_folder,
                    "--extensions-dir", exts_folder,
                    "--install-extension", "ms-toolsai.jupyter",
                    # "--install-extension", "donjayamanne.python-extension-pack"
                ] + ["--install-extension", "ms-python.python@{}".format(python_ext_version)] if python_ext else [],
                env=env,
                stdout=fd,
                stderr=fd,
            )

        if user_folder:
            settings = Path(os.path.expanduser(os.path.join(user_folder, 'User/settings.json')))
            settings.parent.mkdir(parents=True, exist_ok=True)
            # noinspection PyBroadException
            try:
                with open(settings.as_posix(), 'rt') as f:
                    base_json = json.load(f)
            except Exception:
                base_json = {}
            # noinspection PyBroadException
            try:
                base_json.update({
                    "extensions.autoCheckUpdates": False,
                    "extensions.autoUpdate": False,
                    "python.pythonPath": sys.executable,
                    "terminal.integrated.shell.linux": "/bin/bash" if Path("/bin/bash").is_file() else None,
                })
                with open(settings.as_posix(), 'wt') as f:
                    json.dump(base_json, f)
            except Exception:
                pass

        proc = subprocess.Popen(
            ['bash', '-c',
             '{} --auth none --bind-addr 127.0.0.1:{} --disable-update-check {} {}'.format(
                 vscode_path, port,
                 '--user-data-dir \"{}\"'.format(user_folder) if user_folder else '',
                 '--extensions-dir \"{}\"'.format(exts_folder) if exts_folder else '')],
            env=env,
            stdout=fd,
            stderr=fd,
            cwd=cwd,
        )

        try:
            error_code = proc.wait(timeout=1)
            raise ValueError("code-server failed starting, return code {}".format(error_code))
        except subprocess.TimeoutExpired:
            pass

    except Exception as ex:
        print('Failed running vscode server: {}'.format(ex))
        return

    task.set_parameter(name='properties/vscode_port', value=str(port))


def start_jupyter_server(hostname, hostnames, param, task, env):
    if not param.get('jupyterlab', True):
        print('no jupyterlab to monitor - going to sleep')
        while True:
            sleep(10.)
        return  # noqa

    # execute jupyter notebook
    fd, local_filename = mkstemp()
    cwd = (
        os.path.expandvars(os.path.expanduser(param["user_base_directory"]))
        if param["user_base_directory"]
        else os.getcwd()
    )

    # find a free tcp port
    port = get_free_port(8888, 9000)

    # if we are not running as root, make sure the sys executable is in the PATH
    env = dict(**env)
    env['PATH'] = '{}:{}'.format(Path(sys.executable).parent.as_posix(), env.get('PATH', ''))

    # make sure we have the needed cwd
    # noinspection PyBroadException
    try:
        Path(cwd).mkdir(parents=True, exist_ok=True)
    except Exception:
        print("Warning: failed setting user base directory [{}] reverting to ~/".format(cwd))
        cwd = os.path.expanduser("~/")

    print(
        "Running Jupyter Notebook Server on {} [{}] port {} at {}".format(hostname, hostnames, port, cwd)
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            "--no-browser",
            "--allow-root",
            "--ip",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=fd,
        stderr=fd,
        cwd=cwd,
    )
    return monitor_jupyter_server(fd, local_filename, process, task, port, hostnames)


def setup_ssh_server(hostname, hostnames, param, task, env):
    if not param.get("ssh_server"):
        return

    # make sure we do not pass it along to others, work on a copy
    env = deepcopy(env)
    env.pop('LOCAL_PYTHON', None)
    env.pop('PYTHONPATH', None)
    env.pop('DEBIAN_FRONTEND', None)

    print("Installing SSH Server on {} [{}]".format(hostname, hostnames))
    ssh_password = param.get("ssh_password", "training")
    # noinspection PyBroadException
    try:
        ssh_port = param.get("ssh_ports") or "10022:15000"
        min_port = int(ssh_port.split(":")[0])
        max_port = max(min_port+32, int(ssh_port.split(":")[-1]))
        port = get_free_port(min_port, max_port)
        proxy_port = get_free_port(min_port, max_port)
        use_dropbear = False

        # if we are root, install open-ssh
        if os.geteuid() == 0:
            # noinspection SpellCheckingInspection
            os.system(
                "export PYTHONPATH=\"\" && "
                "([ ! -z $(which sshd) ] || (apt-get update ; apt-get install -y openssh-server)) && "
                "mkdir -p /var/run/sshd && "
                "echo 'root:{password}' | chpasswd && "
                "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && "
                "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && "
                "sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd "
                "&& "  # noqa: W605
                "echo 'ClientAliveInterval 10' >> /etc/ssh/sshd_config && "
                "echo 'ClientAliveCountMax 20' >> /etc/ssh/sshd_config && "
                "echo 'AcceptEnv CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY "
                "CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY' >> /etc/ssh/sshd_config && "
                'echo "export VISIBLE=now" >> /etc/profile && '
                'echo "export PATH=$PATH" >> /etc/profile && '
                'echo "ldconfig" 2>/dev/null >> /etc/profile && '
                'echo "export CLEARML_CONFIG_FILE={trains_config_file}" >> /etc/profile'.format(
                    password=ssh_password,
                    port=port,
                    trains_config_file=os.environ.get("CLEARML_CONFIG_FILE") or os.environ.get("CLEARML_CONFIG_FILE"),
                )
            )
            sshd_path = '/usr/sbin/sshd'
            ssh_config_path = '/etc/ssh/'
            custom_ssh_conf = None
        else:
            # check if sshd exists
            # noinspection PyBroadException
            try:
                os.system('echo "export CLEARML_CONFIG_FILE={trains_config_file}" >> $HOME/.profile'.format(
                    trains_config_file=os.environ.get("CLEARML_CONFIG_FILE") or os.environ.get("CLEARML_CONFIG_FILE"),
                ))
            except Exception:
                print("warning failed setting ~/.profile")

            # check if shd is preinstalled
            # noinspection PyBroadException
            try:
                # try running SSHd as non-root (currently bypassed, use dropbear instead)
                sshd_path = None ## subprocess.check_output('which sshd', shell=True).decode().strip()
                if not sshd_path:
                    raise ValueError("sshd was not found")
            except Exception:
                # noinspection PyBroadException
                try:
                    print('SSHd was not found default to user space dropbear sshd server')
                    dropbear_download_link = \
                        os.environ.get("CLEARML_DROPBEAR_EXEC") or \
                        'https://github.com/allegroai/dropbear/releases/download/DROPBEAR_CLEARML_2023.02/dropbearmulti'
                    dropbear = StorageManager.get_local_copy(dropbear_download_link, extract_archive=False)
                    os.chmod(dropbear, 0o744)
                    sshd_path = dropbear
                    use_dropbear = True

                except Exception:
                    print('Error: failed locating SSHd and dailed fetching `dropbear`, leaving!')
                    return

            # noinspection PyBroadException
            try:
                ssh_config_path = os.path.join(os.getcwd(), '.clearml_session_sshd')
                # noinspection PyBroadException
                try:
                    Path(ssh_config_path).mkdir(parents=True, exist_ok=True)
                except Exception:
                    ssh_config_path = os.path.join(gettempdir(), '.clearml_session_sshd')
                    Path(ssh_config_path).mkdir(parents=True, exist_ok=True)

                custom_ssh_conf = os.path.join(ssh_config_path, 'sshd_config')
                with open(custom_ssh_conf, 'wt') as f:
                    conf = \
                        "PermitRootLogin yes" + "\n"\
                        "ClientAliveInterval 10" + "\n"\
                        "ClientAliveCountMax 20" + "\n"\
                        "AllowTcpForwarding yes" + "\n"\
                        "UsePAM yes" + "\n"\
                        "AuthorizedKeysFile {}".format(os.path.join(ssh_config_path, 'authorized_keys')) + "\n"\
                        "PidFile {}".format(os.path.join(ssh_config_path, 'sshd.pid')) + "\n"\
                        "AcceptEnv CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY "\
                        "CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY"+"\n"
                    for k in default_ssh_fingerprint:
                        filename = os.path.join(ssh_config_path, '{}'.format(k.replace('__pub', '.pub')))
                        conf += "HostKey {}\n".format(filename)

                    f.write(conf)
            except Exception:
                print('Error: Cannot configure sshd, leaving!')
                return

            if not use_dropbear:
                # clear the ssh password, we cannot change it
                ssh_password = None
                task.set_parameter('{}/ssh_password'.format(config_section_name), '')

        # create fingerprint files
        Path(ssh_config_path).mkdir(parents=True, exist_ok=True)
        keys_filename = {}
        for k, v in default_ssh_fingerprint.items():
            filename = os.path.join(ssh_config_path, '{}'.format(k.replace('__pub', '.pub')))
            try:
                os.unlink(filename)
            except Exception:  # noqa
                pass
            if v:
                with open(filename, 'wt') as f:
                    f.write(v + (' {}@{}'.format(
                        getpass.getuser() or "root", hostname) if filename.endswith('.pub') else ''))
                os.chmod(filename, 0o600 if filename.endswith('.pub') else 0o600)
            keys_filename[k] = filename

        # run server in foreground so it gets killed with us
        if use_dropbear:
            # convert key files
            dropbear_key_files = []
            for k, ssh_key_file in keys_filename.items():
                # skip over the public keys, there is no need for them
                if ssh_key_file.endswith(".pub"):
                    continue
                drop_key_file = ssh_key_file + ".dropbear"
                try:
                    os.system("{} dropbearconvert openssh dropbear {} {}".format(
                        sshd_path, ssh_key_file, drop_key_file))
                    if Path(drop_key_file).is_file():
                        dropbear_key_files += ["-r", drop_key_file]
                except Exception:
                    pass
            proc_args = [sshd_path, "dropbear", "-e", "-K", "30", "-I", "0", "-F", "-p", str(port)] + dropbear_key_files
            # this is a copy oof `env` so there is nothing to worry about
            env["DROPBEAR_CLEARML_FIXED_PASSWORD"] = ssh_password
        else:
            proc_args = [sshd_path, "-D", "-p", str(port)] + (["-f", custom_ssh_conf] if custom_ssh_conf else [])

        proc = subprocess.Popen(args=proc_args, env=env)
        # noinspection PyBroadException
        try:
            result = proc.wait(timeout=1)
        except Exception:
            result = 0

        if result != 0:
            raise ValueError("Failed launching sshd: ", proc_args)

        # noinspection PyBroadException
        try:
            TcpProxy(listen_port=proxy_port, target_port=port, proxy_state={}, verbose=False,  # noqa
                     keep_connection=True, is_connection_server=True)
        except Exception as ex:
            print('Warning: Could not setup stable ssh port, {}'.format(ex))
            proxy_port = None

        if task:
            if proxy_port:
                task.set_parameter(name='properties/internal_stable_ssh_port', value=str(proxy_port))
            task.set_parameter(name='properties/internal_ssh_port', value=str(port))

        print(
            "\n#\n# SSH Server running on {} [{}] port {}\n# LOGIN u:root p:{}\n#\n".format(
                hostname, hostnames, port, ssh_password
            )
        )

    except Exception as ex:
        print("Error: {}\n\n#\n# Error: SSH server could not be launched\n#\n".format(ex))


def _b64_decode_file(encoded_string):
    # noinspection PyBroadException
    try:
        import gzip
        value = gzip.decompress(base64.decodebytes(encoded_string.encode('ascii'))).decode('utf8')
        return value
    except Exception:
        return None


def setup_user_env(param, task):
    env = setup_os_env(param)

    # apply vault if we have it
    vault_environment = {}
    if param.get("user_key") and param.get("user_secret"):
        # noinspection PyBroadException
        try:
            print('Applying vault configuration')
            from clearml.backend_api.session.defs import ENV_ENABLE_ENV_CONFIG_SECTION, ENV_ENABLE_FILES_CONFIG_SECTION
            prev_env, prev_files = ENV_ENABLE_ENV_CONFIG_SECTION.get(), ENV_ENABLE_FILES_CONFIG_SECTION.get()
            ENV_ENABLE_ENV_CONFIG_SECTION.set(True), ENV_ENABLE_FILES_CONFIG_SECTION.set(True)
            prev_envs = deepcopy(os.environ)
            Session(api_key=param.get("user_key"), secret_key=param.get("user_secret"))
            vault_environment = {k: v for k, v in os.environ.items() if prev_envs.get(k) != v}
            if prev_env is None:
                ENV_ENABLE_ENV_CONFIG_SECTION.pop()
            else:
                ENV_ENABLE_ENV_CONFIG_SECTION.set(prev_env)
            if prev_files is None:
                ENV_ENABLE_FILES_CONFIG_SECTION.pop()
            else:
                ENV_ENABLE_FILES_CONFIG_SECTION.set(prev_files)
            if vault_environment:
                print('Vault environment added: {}'.format(list(vault_environment.keys())))
        except Exception as ex:
            print('Applying vault configuration failed: {}'.format(ex))

    # do not change user bash/profile if we are not running inside a container
    if os.geteuid() != 0:
        # check if we are inside a container
        is_container = False
        try:
            with open("/proc/1/sched", "rt") as f:
                lines = f.readlines()
                if lines and lines[0].split()[0] in ("bash", "sh", "zsh"):
                    # this a container
                    is_container = True
        except Exception:  # noqa
            pass

        if not is_container:
            if param.get("user_key") and param.get("user_secret"):
                env['CLEARML_API_ACCESS_KEY'] = param.get("user_key")
                env['CLEARML_API_SECRET_KEY'] = param.get("user_secret")
            return env

    # create symbolic link to the venv
    environment = os.path.expanduser('~/environment')
    # noinspection PyBroadException
    try:
        os.symlink(os.path.abspath(os.path.join(os.path.abspath(sys.executable), '..', '..')), environment)
        print('Virtual environment are available at {}'.format(environment))
    except Exception:
        pass
    # set default user credentials
    if param.get("user_key") and param.get("user_secret"):
        os.system("echo 'export CLEARML_API_ACCESS_KEY=\"{}\"' >> ~/.bashrc".format(
            param.get("user_key", "").replace('$', '\\$')))
        os.system("echo 'export CLEARML_API_SECRET_KEY=\"{}\"' >> ~/.bashrc".format(
            param.get("user_secret", "").replace('$', '\\$')))
        os.system("echo 'export CLEARML_DOCKER_IMAGE=\"{}\"' >> ~/.bashrc".format(
            param.get("default_docker", "").strip() or env.get('CLEARML_DOCKER_IMAGE', '')))
        os.system("echo 'export CLEARML_API_ACCESS_KEY=\"{}\"' >> ~/.profile".format(
            param.get("user_key", "").replace('$', '\\$')))
        os.system("echo 'export CLEARML_API_SECRET_KEY=\"{}\"' >> ~/.profile".format(
            param.get("user_secret", "").replace('$', '\\$')))
        os.system("echo 'export CLEARML_DOCKER_IMAGE=\"{}\"' >> ~/.profile".format(
            param.get("default_docker", "").strip() or env.get('CLEARML_DOCKER_IMAGE', '')))
        for k, v in vault_environment.items():
            os.system("echo 'export {}=\"{}\"' >> ~/.profile".format(k, v))
            os.system("echo 'export {}=\"{}\"' >> ~/.bashrc".format(k, v))
            env[k] = str(v) if v else ""
        env['CLEARML_API_ACCESS_KEY'] = param.get("user_key")
        env['CLEARML_API_SECRET_KEY'] = param.get("user_secret")
    # set default folder for user
    if param.get("user_base_directory"):
        base_dir = param.get("user_base_directory")
        if ' ' in base_dir:
            base_dir = '\"{}\"'.format(base_dir)
        os.system("echo 'cd {}' >> ~/.bashrc".format(base_dir))
        os.system("echo 'cd {}' >> ~/.profile".format(base_dir))

    # make sure we activate the venv in the bash
    os.system("echo 'source {}' >> ~/.bashrc".format(os.path.join(environment, 'bin', 'activate')))
    os.system("echo '. {}' >> ~/.profile".format(os.path.join(environment, 'bin', 'activate')))

    # check if we need to create .git-credentials

    runtime_property_support = Session.check_min_api_version("2.13")
    if runtime_property_support:
        # noinspection PyProtectedMember
        runtime_prop = dict(task._get_runtime_properties())
        git_credentials = runtime_prop.pop('_git_credentials', None)
        git_config = runtime_prop.pop('_git_config', None)
        # force removing properties
        # noinspection PyProtectedMember
        task._edit(runtime=runtime_prop)
        task.reload()
        if git_credentials is not None:
            git_credentials = _b64_decode_file(git_credentials)
        if git_config is not None:
            git_config = _b64_decode_file(git_config)
    else:
        # noinspection PyProtectedMember
        git_credentials = task._get_configuration_text('git_credentials')
        # noinspection PyProtectedMember
        git_config = task._get_configuration_text('git_config')

    if git_credentials:
        git_cred_file = os.path.expanduser('~/.config/git/credentials')
        # noinspection PyBroadException
        try:
            Path(git_cred_file).parent.mkdir(parents=True, exist_ok=True)
            with open(git_cred_file, 'wt') as f:
                f.write(git_credentials)
        except Exception:
            print('Could not write {} file'.format(git_cred_file))

    if git_config:
        git_config_file = os.path.expanduser('~/.config/git/config')
        # noinspection PyBroadException
        try:
            Path(git_config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(git_config_file, 'wt') as f:
                f.write(git_config)
        except Exception:
            print('Could not write {} file'.format(git_config_file))

    return env


def get_host_name(task, param):
    # noinspection PyBroadException
    try:
        hostname = socket.gethostname()
        hostnames = socket.gethostbyname(socket.gethostname())
    except Exception:
        def get_ip_addresses(family):
            for interface, snics in psutil.net_if_addrs().items():
                for snic in snics:
                    if snic.family == family:
                        yield snic.address

        hostnames = list(get_ip_addresses(socket.AF_INET))[0]
        hostname = hostnames

    # try to get external address (if possible)
    # noinspection PyBroadException
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # noinspection PyBroadException
        try:
            # doesn't even have to be reachable
            s.connect(('8.255.255.255', 1))
            hostnames = s.getsockname()[0]
        except Exception:
            pass
        finally:
            s.close()
    except Exception:
        pass

    # update host name
    if not task.get_parameter(name='properties/external_address'):
        external_addr = hostnames
        if param.get('public_ip'):
            # noinspection PyBroadException
            try:
                external_addr = requests.get('https://checkip.amazonaws.com').text.strip()
            except Exception:
                pass
        task.set_parameter(name='properties/external_address', value=str(external_addr))

    return hostname, hostnames


def run_user_init_script(task):
    # run initialization script:
    # noinspection PyProtectedMember
    init_script = task._get_configuration_text(config_object_section_bash_init)
    if not init_script or not str(init_script).strip():
        return
    print("Running user initialization bash script:")
    init_filename = os_json_filename = None
    try:
        fd, init_filename = mkstemp(suffix='.init.sh')
        os.close(fd)
        fd, os_json_filename = mkstemp(suffix='.env.json')
        os.close(fd)
        with open(init_filename, 'wt') as f:
            f.write(init_script +
                    '\n{} -c '
                    '"exec(\\"try:\\n import os\\n import json\\n'
                    ' json.dump(dict(os.environ), open(\\\'{}\\\', \\\'w\\\'))'
                    '\\nexcept: pass\\")"'.format(sys.executable, os_json_filename))
        env = dict(**os.environ)
        # do not pass or update back the PYTHONPATH environment variable
        env.pop('PYTHONPATH', None)
        subprocess.call(['/bin/bash', init_filename], env=env)
        with open(os_json_filename, 'rt') as f:
            environ = json.load(f)
        # do not pass or update back the PYTHONPATH environment variable
        environ.pop('PYTHONPATH', None)
        # update environment variables
        os.environ.update(environ)
    except Exception as ex:
        print('User initialization script failed: {}'.format(ex))
    finally:
        if init_filename:
            try:
                os.unlink(init_filename)
            except:  # noqa
                pass
        if os_json_filename:
            try:
                os.unlink(os_json_filename)
            except:  # noqa
                pass
    os.environ['CLEARML_DOCKER_BASH_SCRIPT'] = str(init_script)


def main():
    param = {
        "user_base_directory": "~/",
        "ssh_server": True,
        "ssh_password": "training",
        "default_docker": "nvidia/cuda",
        "user_key": None,
        "user_secret": None,
        "vscode_server": True,
        "vscode_version": '',
        "jupyterlab": True,
        "public_ip": False,
        "ssh_ports": None,
    }
    task = init_task(param, default_ssh_fingerprint)

    run_user_init_script(task)

    hostname, hostnames = get_host_name(task, param)

    env = setup_user_env(param, task)

    setup_ssh_server(hostname, hostnames, param, task, env)

    start_vscode_server(hostname, hostnames, param, task, env)

    start_jupyter_server(hostname, hostnames, param, task, env)

    print('We are done')


if __name__ == '__main__':
    main()
