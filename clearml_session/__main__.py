import hashlib
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, FileType
from functools import reduce
from io import TextIOBase
from time import time, sleep

if sys.platform == 'win32':
    import msvcrt  # noqa
    import wexpect as pexpect  # noqa
else:
    import select  # noqa
    import pexpect  # noqa

import psutil
from clearml import Task
from clearml.backend_api.session.client import APIClient
from clearml.config import config_obj
from .tcp_proxy import TcpProxy
from .single_thread_proxy import SingleThreadProxy


system_tag = 'interactive'
default_docker_image = 'nvidia/cuda:10.1-runtime-ubuntu18.04'


def _read_std_input(timeout):
    # wait for user input with timeout, return None if timeout or user input
    if sys.platform == 'win32':
        start_time = time()
        input_str = ''
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getche()
                if ord(char) == 13:  # enter_key
                    print('')
                    return input_str.strip()
                input_str += char.decode()
            if len(input_str) == 0 and (time() - start_time) > timeout:
                return None
    else:
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if not i:
            return None
        line = sys.stdin.readline().strip()
        # flush stdin buffer
        while i:
            i, o, e = select.select([sys.stdin], [], [], 0)
            if i:
                sys.stdin.readline()
        return line


def _get_config_section_name():
    org_path = [p for p in sys.path]
    # noinspection PyBroadException
    try:
        sys.path.append(os.path.abspath(os.path.join(__file__, '..',)))
        from interactive_session_task import (  # noqa
            config_section_name, config_object_section_ssh, config_object_section_bash_init)  # noqa
        return config_section_name, config_object_section_ssh, config_object_section_bash_init
    except Exception:
        return None, None, None
    finally:
        sys.path = org_path


def _check_ssh_executable():
    # check Windows 32bit version is not supported
    if sys.platform == 'win32' and getattr(sys, 'winver', '').endswith('-32'):
        raise ValueError("Python 32-bit version detected. Only Python 64-bit is supported!")

    # noinspection PyBroadException
    try:
        if sys.platform == 'win32':
            ssh_exec = subprocess.check_output('where ssh.exe'.split()).decode('utf-8').split('\n')[0].strip()
        else:
            ssh_exec = subprocess.check_output('which ssh'.split()).decode('utf-8').split('\n')[0].strip()
        return ssh_exec
    except Exception:
        return None


def create_base_task(state, project_name=None, task_name=None):
    task = Task.create(project_name=project_name or 'DevOps',
                       task_name=task_name or 'Interactive Session',
                       task_type=Task.TaskTypes.application)
    task_state = task.export_task()
    base_script_file = os.path.abspath(os.path.join(__file__, '..', 'tcp_proxy.py'))
    with open(base_script_file, 'rt') as f:
        task_state['script']['diff'] = f.read()
    base_script_file = os.path.abspath(os.path.join(__file__, '..', 'interactive_session_task.py'))
    with open(base_script_file, 'rt') as f:
        task_state['script']['diff'] += '\n\n' + f.read()

    task_state['script']['working_dir'] = '.'
    task_state['script']['entry_point'] = 'interactive_session.py'
    task_state['script']['requirements'] = {'pip': '\n'.join(
        ["clearml", "jupyter", "jupyterlab", "jupyterlab_git"] +
        (['pylint'] if state.get('vscode_server') else []))}
    task.update_task(task_state)
    section, _, _ = _get_config_section_name()
    task.set_parameters({
        "{}/user_base_directory".format(section): "~/",
        "{}/ssh_server".format(section): True,
        "{}/ssh_password".format(section): "training",
        "{}/default_docker".format(section): "nvidia/cuda",
        "{}/user_key".format(section): '',
        "{}/user_secret".format(section): '',
        "properties/external_address": '',
        "properties/internal_ssh_port": 10022,
        "properties/jupyter_token": '',
        "properties/jupyter_port": '',
    })
    task.set_system_tags([system_tag])
    task.reset(force=True)
    return task


def create_debugging_task(state, debug_task_id):
    debug_task = Task.get_task(task_id=debug_task_id)
    # if there is no git repository, we cannot debug it
    if not debug_task.data.script.repository:
        raise ValueError("Debugging task has no git repository, single script debugging is not supported.")

    task = Task.clone(source_task=debug_task_id, parent=debug_task_id)

    task_state = task.export_task()

    base_script_file = os.path.abspath(os.path.join(__file__, '..', 'interactive_session_task.py'))
    with open(base_script_file, 'rt') as f:
        entry_diff = ['+'+line.rstrip() for line in f.readlines()]
    entry_diff_header = \
        "diff --git a/__interactive_session__.py b/__interactive_session__.py\n" \
        "--- a/__interactive_session__.py\n" \
        "+++ b/__interactive_session__.py\n" \
        "@@ -0,0 +1,{} @@\n".format(len(entry_diff))

    task_state['script']['diff'] = \
        entry_diff_header + '\n'.join(entry_diff) + '\n' + (task_state['script']['diff'] or '')
    task_state['script']['working_dir'] = '.'
    task_state['script']['entry_point'] = '__interactive_session__.py'
    state['packages'] = (state.get('packages') or []) + ["clearml", "jupyter", "jupyterlab", "jupyterlab_git"] + (
        ['pylint'] if state.get('vscode_server') else [])
    task.update_task(task_state)
    section, _, _ = _get_config_section_name()
    task.set_parameters({
        "{}/user_base_directory".format(section): "~/",
        "{}/ssh_server".format(section): True,
        "{}/ssh_password".format(section): "training",
        "{}/default_docker".format(section): "nvidia/cuda",
        "{}/user_key".format(section): '',
        "{}/user_secret".format(section): '',
        "properties/external_address": '',
        "properties/internal_ssh_port": 10022,
        "properties/jupyter_token": '',
        "properties/jupyter_port": '',
    })
    task.set_system_tags([system_tag])
    task.reset(force=True)
    return task


def delete_old_tasks(client, base_task_id):
    print('Removing stale interactive sessions')
    res = client.session.send_request(service='users', action='get_current_user', async_enable=False)
    assert res.ok
    current_user_id = res.json()['data']['user']['id']
    previous_tasks = client.tasks.get_all(**{
        'status': ['failed', 'stopped', 'completed'],
        'parent': base_task_id or None,
        'system_tags': None if base_task_id else [system_tag],
        'page_size': 100, 'page': 0,
        'user': [current_user_id], 'only_fields': ['id']
    })
    for t in previous_tasks:
        try:
            client.tasks.delete(task=t.id, force=True)
        except Exception as ex:
            logging.getLogger().warning('{}\nFailed deleting old session {}'.format(ex, t.id))


def get_project_id(state):
    project_id = None
    project_name = state.get('project') or None
    if project_name:
        projects = Task.get_projects()
        project_id = [p for p in projects if p.name == project_name]
        if project_id:
            project_id = project_id[0]
        else:
            logging.getLogger().warning("could not locate project by the named '{}'".format(project_name))
            project_id = None
    return project_id


def get_user_inputs(args, parser, state, client):
    default_needed_args = tuple()

    user_args = sorted([a for a in args.__dict__ if not a.startswith('_')])
    # clear some states if we replace the base_task_id
    if 'base_task_id' in user_args and getattr(args, 'base_task_id', None) != state.get('base_task_id'):
        print('New base_task_id \'{}\', clearing previous packages & init_script'.format(
            getattr(args, 'base_task_id', None)))
        state.pop('init_script', None)
        state.pop('packages', None)
        state.pop('base_task_id', None)

    if str(getattr(args, 'base_task_id', '')).lower() == 'none':
        args.base_task_id = None
        state['base_task_id'] = None

    for a in user_args:
        v = getattr(args, a, None)
        if a in ('requirements', 'packages', 'attach', 'config_file'):
            continue
        if isinstance(v, TextIOBase):
            state[a] = v.read()
        elif not v and a == 'init_script':
            if v is None:
                state[a] = ''
            else:
                pass  # keep as is
        elif not v and a == 'remote_gateway':
            state.pop(a, None)
        elif v is not None:
            state[a] = v

        if a in default_needed_args and not state.get(a):
            # noinspection PyProtectedMember
            state[a] = input(
                "\nCould not locate previously used value of '{}', please provide it?"
                "\n    Help: {}\n> ".format(
                    a, parser._option_string_actions['--{}'.format(a.replace('_', '-'))].help))
    # if no password was set, create a new random one
    if not state.get('password'):
        state['password'] = hashlib.sha256("seed me Seymour {}".format(time()).encode()).hexdigest()

    # store the requirements from the requirements.txt
    # override previous requirements
    if args.requirements:
        state['packages'] = (args.packages or []) + [
            p.strip() for p in args.requirements.readlines() if not p.strip().startswith('#')]
    elif args.packages is not None:
        state['packages'] = args.packages or []

    # allow to select queue
    ask_queues = not state.get('queue')
    if state.get('queue'):
        choice = input('Use previous queue (resource) \'{}\' [Y]/n? '.format(state['queue']))
        if choice in ('n', 'N', 'no', 'No', 'NO'):
            ask_queues = True
    if ask_queues:
        print('Select the queue (resource) you request:')
        queues = sorted([q.name for q in client.queues.get_all(
            system_tags=['-{}'.format(t) for t in state.get('queue_excluded_tag', ['internal'])] +
                        ['{}'.format(t) for t in state.get('queue_include_tag', [])])])
        queues_list = '\n'.join('{}] {}'.format(i, q) for i, q in enumerate(queues))
        while True:
            try:
                choice = int(input(queues_list+'\nSelect a queue [0-{}] '.format(len(queues))))
                break
            except (TypeError, ValueError):
                pass
        state['queue'] = queues[int(choice)]

    print("\nInteractive session config:\n{}\n".format(
        json.dumps({k: v for k, v in state.items() if not str(k).startswith('__')}, indent=4, sort_keys=True)))

    choice = input('Launch interactive session [Y]/n? ')
    if choice in ('n', 'N', 'no', 'No', 'NO'):
        print('User aborted')
        exit(0)

    return state


def save_state(state, state_file):
    # if we are running in debugging mode,
    # only store the current task (do not change the defaults)
    if state.get('debugging'):
        # noinspection PyBroadException
        base_state = load_state(state_file)
        base_state['task_id'] = state.get('task_id')
        state = base_state

    state['__version__'] = get_version()
    # save new state
    with open(state_file, 'wt') as f:
        json.dump(state, f, sort_keys=True)


def load_state(state_file):
    # noinspection PyBroadException
    try:
        with open(state_file, 'rt') as f:
            state = json.load(f)
    except Exception:
        state = {}
    return state


def clone_task(state, project_id):
    new_task = False
    if state.get('debugging'):
        print('Starting new debugging session to {}'.format(state.get('debugging')))
        task = create_debugging_task(state, state.get('debugging'))
    elif state.get('base_task_id'):
        print('Cloning base session {}'.format(state['base_task_id']))
        task = Task.clone(source_task=state['base_task_id'], project=project_id, parent=state['base_task_id'])
        task.set_system_tags([system_tag])
    else:
        print('Creating new session')
        task = create_base_task(state, project_name=state.get('project'))
        new_task = True

    task_params = task.get_parameters(backwards_compatibility=False)
    if 'General/ssh_server' in task_params:
        section = 'General'
        init_section = 'init_script'
    else:
        section, _, init_section = _get_config_section_name()
    task_params['properties/jupyter_token'] = ''
    task_params['properties/jupyter_port'] = ''
    if state.get('remote_gateway') is not None:
        task_params['properties/external_address'] = str(state.get('remote_gateway'))
    task_params['{}/ssh_server'.format(section)] = str(True)
    task_params['{}/ssh_password'.format(section)] = state['password']
    task_params['{}/user_key'.format(section)] = config_obj.get("api.credentials.access_key")
    task_params['{}/user_secret'.format(section)] = config_obj.get("api.credentials.secret_key")
    task_params["{}/vscode_server".format(section)] = bool(state.get('vscode_server'))
    task_params["{}/public_ip".format(section)] = bool(state.get('public_ip'))
    if state.get('user_folder'):
        task_params['{}/user_base_directory'.format(section)] = state.get('user_folder')
    docker = state.get('docker') or task.data.execution.docker_cmd
    if not state.get('skip_docker_network') and not docker:
        docker = default_docker_image
    if docker:
        task_params['{}/default_docker'.format(section)] = docker.replace('--network host', '').strip()
        task.set_base_docker(docker + (
            ' --network host' if not state.get('skip_docker_network') and '--network host' not in docker else ''))
    # set the bash init script
    if state.get('init_script') is not None and (not new_task or state.get('init_script').strip()):
        # noinspection PyProtectedMember
        task._set_configuration(name=init_section, config_type='bash', config_text=state.get('init_script') or '')

    # store the .git-credentials
    if state.get('git_credentials'):
        git_cred_file = os.path.join(os.path.expanduser('~'), '.git-credentials')
        if os.path.isfile(git_cred_file):
            task.connect_configuration(
                configuration=git_cred_file, name='git_credentials', description='git credentials')
        git_conf_file = os.path.join(os.path.expanduser('~'), '.gitconfig')
        if os.path.isfile(git_conf_file):
            task.connect_configuration(
                configuration=git_conf_file, name='git_config', description='git config')

    if state.get('packages'):
        requirements = task.data.script.requirements or {}
        # notice split order is important!
        packages = [p for p in state['packages'] if p.strip() and not p.strip().startswith('#')]
        packages_id = set(reduce(lambda a, b: a.split(b)[0], "#;@=~<>", p).strip() for p in packages)
        if isinstance(requirements.get('pip'), str):
            requirements['pip'] = requirements['pip'].split('\n')
        for p in (requirements.get('pip') or []):
            if not p.strip() or p.strip().startswith('#'):
                continue
            p_id = reduce(lambda a, b: a.split(b)[0], "#;@=~<>", p).strip()
            if p_id not in packages_id:
                packages += [p]

        requirements['pip'] = '\n'.join(sorted(packages))
        task.update_task({'script': {'requirements': requirements}})
    task.set_parameters(task_params)
    print('New session created [id={}]'.format(task.id))
    return task


def wait_for_machine(task):
    # wait until task is running
    print('Waiting for remote machine allocation [id={}]'.format(task.id))
    last_status = None
    while last_status != 'in_progress' and last_status in (None, 'created', 'queued', 'unknown',):
        print('.', end='', flush=True)
        if last_status is not None:
            sleep(2.)
        status = task.get_status()
        if last_status != status:
            # noinspection PyProtectedMember
            last_status = task._get_status()[1]
            print('Status [{}]{}'.format(status, ' - {}'.format(last_status) if last_status else ''))
        last_status = status
    print('Remote machine allocated')
    print('Setting remote environment [Task id={}]'.format(task.id))
    print('Setup process details: {}'.format(task.get_output_log_web_page()))
    print('Waiting for environment setup to complete [usually about 20-30 seconds]')
    # monitor progress, until we get the new jupyter, then we know it is working
    task.reload()
    while not task.get_parameter('properties/jupyter_port') and task.get_status() == 'in_progress':
        print('.', end='', flush=True)
        sleep(3.)
        task.reload()
    if task.get_status() != 'in_progress':
        raise ValueError("Remote setup failed (status={}) see details: {}".format(
            task.get_status(), task.get_output_log_web_page()))
    print('\nRemote machine is ready')

    return task


def start_ssh_tunnel(remote_address, ssh_port, ssh_password, local_remote_pair_list):
    print('Starting SSH tunnel')
    child = None
    args = ['-N', '-C',
            'root@{}'.format(remote_address), '-p', '{}'.format(ssh_port),
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=10', ]

    for local, remote in local_remote_pair_list:
        args.extend(['-L', '{}:localhost:{}'.format(local, remote)])

    # noinspection PyBroadException
    try:
        child = pexpect.spawn(
            command=_check_ssh_executable(),
            args=args,
            logfile=sys.stdout, timeout=20, encoding='utf-8')
        i = child.expect(['password:', r'\(yes\/no\)', r'.*[$#] ', pexpect.EOF])
        if i == 0:
            child.sendline(ssh_password)
            try:
                child.expect(['password:'], timeout=5)
                print('Incorrect password')
                raise ValueError('Incorrect password')
            except pexpect.TIMEOUT:
                pass

        elif i == 1:
            child.sendline("yes")
            ret1 = child.expect(["password:", pexpect.EOF])
            if ret1 == 0:
                child.sendline(ssh_password)
                try:
                    child.expect(['password:'], timeout=5)
                    print('Incorrect password')
                    raise ValueError('Incorrect password')
                except pexpect.TIMEOUT:
                    pass
    except Exception:
        child.terminate(force=True)
        child = None
    print('\n')
    return child


def monitor_ssh_tunnel(state, task):
    print('Setting up connection to remote session')
    local_jupyter_port = 8878
    local_ssh_port = 8022
    local_vscode_port = 8898
    ssh_process = None
    sleep_period = 3
    ssh_port = jupyter_token = jupyter_port = internal_ssh_port = ssh_password = remote_address = None
    vscode_port = None

    connect_state = {'reconnect': False}
    if not state.get('disable_keepalive'):
        local_jupyter_port_ = local_jupyter_port + 1
        SingleThreadProxy(local_jupyter_port, local_jupyter_port_)
        local_vscode_port_ = local_vscode_port + 1
        if state.get('vscode_server'):
            SingleThreadProxy(local_vscode_port, local_vscode_port_)
        local_ssh_port_ = local_ssh_port + 1
        TcpProxy(local_ssh_port, local_ssh_port_, connect_state, verbose=False,
                 keep_connection=True, is_connection_server=False)
    else:
        local_jupyter_port_ = local_jupyter_port
        local_ssh_port_ = local_ssh_port
        local_vscode_port_ = local_vscode_port

    default_section = _get_config_section_name()[0]
    local_remote_pair_list = []
    try:
        while task.get_status() == 'in_progress':
            if not all([ssh_port, jupyter_token, jupyter_port, internal_ssh_port, ssh_password, remote_address]):
                task.reload()
                task_parameters = task.get_parameters()
                section = 'General' if 'General/ssh_server' in task_parameters else default_section
                remote_address = \
                    task_parameters.get('properties/k8s-gateway-address') or \
                    task_parameters.get('properties/external_address')
                ssh_password = task_parameters.get('{}/ssh_password'.format(section)) or state['password']
                internal_ssh_port = task_parameters.get('properties/internal_ssh_port')
                jupyter_port = task_parameters.get('properties/jupyter_port')
                jupyter_token = task_parameters.get('properties/jupyter_token')
                ssh_port = \
                    task_parameters.get('properties/k8s-pod-port') or \
                    task_parameters.get('properties/external_ssh_port') or internal_ssh_port
                if not state.get('disable_keepalive'):
                    internal_ssh_port = task_parameters.get('properties/internal_stable_ssh_port') or internal_ssh_port
                local_remote_pair_list = [(local_jupyter_port_, jupyter_port), (local_ssh_port_, internal_ssh_port)]
                if state.get('vscode_server'):
                    vscode_port = task_parameters.get('properties/vscode_port')
                if vscode_port:
                    local_remote_pair_list += [(local_vscode_port_, vscode_port)]

            if not jupyter_port:
                print('Waiting for Jupyter server...')
                continue

            if connect_state.get('reconnect'):
                # noinspection PyBroadException
                try:
                    ssh_process.close(**({'force': True} if sys.platform != 'win32' else {}))
                    ssh_process = None
                except Exception:
                    pass

            if not ssh_process or not ssh_process.isalive():
                ssh_process = start_ssh_tunnel(
                    remote_address, ssh_port, ssh_password,
                    local_remote_pair_list=local_remote_pair_list)

                if ssh_process and ssh_process.isalive():
                    msg = \
                        'Interactive session is running:\n'\
                        'SSH: ssh root@localhost -p {local_ssh_port} [password: {ssh_password}]\n'\
                        'Jupyter Lab URL: http://localhost:{local_jupyter_port}/?token={jupyter_token}'.format(
                            local_jupyter_port=local_jupyter_port, local_ssh_port=local_ssh_port,
                            ssh_password=ssh_password, jupyter_token=jupyter_token)
                    if vscode_port:
                        msg += 'VSCode server available at http://localhost:{local_vscode_port}/'.format(
                            local_vscode_port=local_vscode_port)
                    print(msg)

                    print('\nConnection is up and running\n'
                          'Enter \"r\" (or \"reconnect\") to reconnect the session (for example after suspend)\n'
                          'Ctrl-C (or "quit") to abort (remote session remains active)\n'
                          'or \"Shutdown\" to shutdown remote interactive session')
                else:
                    logging.getLogger().warning('SSH tunneling failed, retrying in {} seconds'.format(3))
                    sleep(3.)
                    continue

            connect_state['reconnect'] = False

            # wait for user input
            user_input = _read_std_input(timeout=sleep_period)
            if user_input is None:
                # noinspection PyBroadException
                try:
                    # check open connections
                    proc = psutil.Process(ssh_process.pid)
                    open_ports = [p.laddr.port for p in proc.connections(kind='tcp4') if p.status == 'LISTEN']
                    remote_ports = [p.raddr.port for p in proc.connections(kind='tcp4') if p.status == 'ESTABLISHED']
                    if int(local_jupyter_port_) not in open_ports or \
                            int(local_ssh_port_) not in open_ports or \
                            int(ssh_port) not in remote_ports:
                        connect_state['reconnect'] = True
                except Exception:
                    pass
                continue

            if user_input.lower() == 'shutdown':
                print('Shutting down interactive session')
                task.mark_stopped()
                break
            elif user_input.lower() in ('r', 'reconnect', ):
                print('Reconnecting to interactive session')
                # noinspection PyBroadException
                try:
                    ssh_process.close(**({'force': True} if sys.platform != 'win32' else {}))
                except Exception:
                    pass
            elif user_input.lower() in ('q', 'quit',):
                raise KeyboardInterrupt()
            else:
                print('unknown command: \'{}\''.format(user_input))

        print('Interactive session ended')
    except KeyboardInterrupt:
        print('\nUser aborted')

    # kill the ssh process
    # noinspection PyBroadException
    try:
        ssh_process.close(**({'force': True} if sys.platform != 'win32' else {}))
    except Exception:
        pass
    # noinspection PyBroadException
    try:
        ssh_process.kill(9 if sys.platform != 'win32' else 15)
    except Exception:
        pass


def setup_parser(parser):
    parser.add_argument('--version', action='store_true', default=None,
                        help='Display the clearml-session utility version')
    parser.add_argument('--attach', default=False, nargs='?',
                        help='Attach to running interactive session (default: previous session)')
    parser.add_argument('--debugging', type=str, default=None,
                        help='Pass existing Task id (experiment), create a copy of the experiment on a remote machine, '
                             'and launch jupyter/ssh for interactive access. Example --debugging <task_id>')
    parser.add_argument('--queue', type=str, default=None,
                        help='Select the queue to launch the interactive session on (default: previously used queue)')
    parser.add_argument('--docker', type=str, default=None,
                        help='Select the docker image to use in the interactive session on '
                             '(default: previously used docker image or `{}`)'.format(default_docker_image))
    parser.add_argument('--public-ip', default=None, nargs='?', const='true', metavar='true/false',
                        type=lambda x: (str(x).strip().lower() in ('true', 'yes')),
                        help='If True register the public IP of the remote machine. Set if running on the cloud. '
                             'Default: false (use for local / on-premises)')
    parser.add_argument('--vscode-server', default=True, nargs='?', const='true', metavar='true/false',
                        type=lambda x: (str(x).strip().lower() in ('true', 'yes')),
                        help='Installing vscode server (code-server) on interactive session (default: true)')
    parser.add_argument('--git-credentials', default=False, nargs='?', const='true', metavar='true/false',
                        type=lambda x: (str(x).strip().lower() in ('true', 'yes')),
                        help='If true, local .git-credentials file is sent to the interactive session. '
                             '(default: false)')
    parser.add_argument('--user-folder', type=str, default=None,
                        help='Advanced: Set the remote base folder (default: ~/)')
    parser.add_argument('--packages', type=str, nargs='*',
                        help='Additional packages to add, supports version numbers '
                             '(default: previously added packages). '
                             'examples: --packages torch==1.7 tqdm')
    parser.add_argument('--requirements', type=FileType('r'), default=None,
                        help='Specify requirements.txt file to install when setting the interactive session. '
                             'Requirements file is read and stored in `packages` section as default for '
                             'the next sessions. Can be overridden by calling `--packages`')
    parser.add_argument('--init-script', type=FileType('r'), default=False, nargs='?',
                        help='Specify BASH init script file to be executed when setting the interactive session. '
                             'Script content is read and stored as default script for the next sessions. '
                             'To clear the init-script do not pass a file')
    parser.add_argument('--config-file', type=str, default='~/.clearml_session.json',
                        help='Advanced: Change the configuration file used to store the previous state '
                             '(default: ~/.clearml_session.json')
    parser.add_argument('--remote-gateway', default=None, nargs='?',
                        help='Advanced: Specify gateway ip/address to be passed to interactive session '
                             '(for use with k8s ingestion / ELB')
    parser.add_argument('--base-task-id', type=str, default=None,
                        help='Advanced: Set the base task ID for the interactive session. '
                             '(default: previously used Task). Use `none` for the default interactive session')
    parser.add_argument('--project', type=str, default=None,
                        help='Advanced: Set the project name for the interactive session Task')
    parser.add_argument('--disable-keepalive', action='store_true', default=None,
                        help='Advanced: If set, disable the transparent proxy always keeping the sockets alive. '
                             'Default: false, use transparent socket mitigating connection drops.')
    parser.add_argument('--queue-excluded-tag', default=None, nargs='*',
                        help='Advanced: Excluded queues with this specific tag from the selection')
    parser.add_argument('--queue-include-tag', default=None, nargs='*',
                        help='Advanced: Only include queues with this specific tag from the selection')
    parser.add_argument('--skip-docker-network', action='store_true', default=None,
                        help='Advanced: If set, `--network host` is **not** passed to docker '
                             '(assumes k8s network ingestion) (default: false)')
    parser.add_argument('--password', type=str, default=None,
                        help='Advanced: Select ssh password for the interactive session '
                             '(default: previously used one)')


def get_version():
    from .version import __version__
    return __version__


def cli():
    title = 'clearml-session - CLI for launching JupyterLab / VSCode on a remote machine'
    print(title)
    parser = ArgumentParser(
        prog='clearml-session', description=title,
        epilog='Notice! all arguments are stored as new defaults for the next session')
    setup_parser(parser)

    # get the args
    args = parser.parse_args()

    if args.version:
        print('Version {}'.format(get_version()))
        exit(0)

    # check ssh
    if not _check_ssh_executable():
        raise ValueError("Could not locate SSH executable")

    # load previous state
    state_file = os.path.abspath(os.path.expandvars(os.path.expanduser(args.config_file)))
    state = load_state(state_file)

    task = None
    if not args.debugging and (args.attach or state.get('task_id')):
        task_id = args.attach or state.get('task_id')
        print('Checking previous session')
        try:
            task = Task.get_task(task_id=task_id)
        except ValueError:
            task = None
        previous_status = task.get_status() if task else None
        if previous_status == 'in_progress':
            # only ask if we were not requested directly
            if args.attach is False:
                choice = input('Connect to active session id={} [Y]/n? '.format(task_id))
                if choice in ('n', 'N', 'no', 'No', 'NO'):
                    task = None
            else:
                print('Using active session id={}'.format(task_id))
        else:
            print('Previous session is unavailable [status={}], starting a new session.'.format(previous_status))
            task = None

    if task:
        state['task_id'] = task.id
        save_state(state, state_file)
    else:
        state.pop('task_id', None)
        save_state(state, state_file)

        print('Verifying credentials')
        client = APIClient()

        # update state with new args
        # and make sure we have all the required fields
        state = get_user_inputs(args, parser, state, client)

        # save state
        save_state(state, state_file)

        # get project name
        project_id = get_project_id(state)

        # remove old Tasks created by us.
        delete_old_tasks(client, state.get('base_task_id'))

        # Clone the Task and adjust parameters
        task = clone_task(state, project_id)
        state['task_id'] = task.id
        save_state(state, state_file)

        # launch
        Task.enqueue(task=task, queue_name=state['queue'])

    # wait for machine to become available
    try:
        wait_for_machine(task)
    except ValueError as ex:
        print('\nERROR: {}'.format(ex))
        return 1

    # launch ssh tunnel
    monitor_ssh_tunnel(state, task)

    # we are done
    print('Leaving interactive session')


def main():
    try:
        cli()
    except KeyboardInterrupt:
        print('\nUser aborted')
    except Exception as ex:
        print('\nError: {}'.format(ex))
        exit(1)


if __name__ == '__main__':
    main()
