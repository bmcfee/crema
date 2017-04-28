#!/usr/bin/env python
'''CREMA utilities'''

import os
import subprocess


def git_version():
    '''Return the git revision as a string

    Returns
    -------
    git_version : str
        The current git revision
    '''
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def increment_version(filename):
    '''Increment a model version identifier.

    Parameters
    ----------
    filename : str
        The file containing the model version

    Returns
    -------
    model_version : str
        The new model version.
        This version will also be written out to `filename`.
    '''

    gv = git_version()
    iteration = 0

    try:
        with open(filename, 'r') as fd:
            line = fd.read()
            old_gv, old_iteration = line.split('.', 2)
            old_iteration = int(old_iteration)

            if old_gv == gv:
                iteration = old_iteration + 1
    except FileNotFoundError:
        pass

    version = '{}.{}'.format(gv, iteration)

    with open(filename, 'w') as fd:
        fd.write(version)

    return version
