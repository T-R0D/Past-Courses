# This file is part of JST.
#
# JST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JST.  If not, see <http://www.gnu.org/licenses/>.

import sys


class Logger(object):
    INFO = 'INFO'
    WARNING = 'WARNING'
    TOKEN = 'TOKEN'
    PRODUCTION = 'PRODUCTION'
    SOURCE = 'SOURCE'
    SYMBOL_TABLE = 'SYMBOL TABLE'

    def __init__(self, file=sys.stdout, switches=None):
        if switches is None:
            switches = {}
        self.switches = switches
        self.file = file

    def add_switch(self, switch, level=0):
        self.switches[switch] = level

    def remove_switch(self, switch):
        self.switches.pop(switch, 0)

    def log(self, switch, message, level=0):
        if switch in self.switches.keys():
            if level <= self.switches.get(switch, 0):
                self.file.write(switch + ': ' + message + '\n')

    def info(self, message, level=0):
        self.log(Logger.INFO, message, level)

    def warning(self, message, level=0):
        self.log(Logger.WARNING, message, level)

    def production(self, message, level=0):
        self.log(Logger.PRODUCTION, message, level)

    def token(self, message, level=0):
        self.log(Logger.TOKEN, message, level)

    def source(self, message, line=-1, level=0):
        message = message if line < 0 else 'line {}: {}'.format(line, message)
        self.log(Logger.SOURCE, message, level)

    def symbol_table(self, message, level=0):
        self.log(Logger.SYMBOL_TABLE, message, level)

    def move_to_file(self, file):
        self.finalize()
        self.file = file

    def finalize(self):
        if self.file not in {sys.stdout, sys.stderr}:
            self.file.close()
        self.file = None
