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


class GenericTicketCounter(object):
    def __init__(self, prefix=''):
        if prefix == '':
            self.prefix = ''
        else:
            self.prefix = prefix + '_'
        self.next_value = 0

    def get(self, count_by=1):
        ticket = '{prefix}{value:>05}'.format(prefix=self.prefix, value=self.next_value)
        self.next_value += count_by
        return ticket

UUID_TICKETS = GenericTicketCounter()
LABEL_TICKETS = GenericTicketCounter(prefix='label')
INT_REGISTER_TICKETS = GenericTicketCounter(prefix='ireg')
FLOAT_REGISTER_TICKETS = GenericTicketCounter(prefix='freg')
LOOP_CONDITION_TICKETS = GenericTicketCounter(prefix='LOOP_CONDITION')
LOOP_BODY_TICKETS = GenericTicketCounter(prefix='LOOP_BODY')
LOOP_EXIT_TICKETS = GenericTicketCounter(prefix='LOOP_EXIT')
IF_TRUE_TICKETS = GenericTicketCounter(prefix='IF_TRUE')
IF_FALSE_TICKETS = GenericTicketCounter(prefix='IF_FALSE')
ENDIF_TICKETS = GenericTicketCounter(prefix='ENDIF')