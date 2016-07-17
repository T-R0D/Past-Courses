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

from ticket_counting.ticket_counters import UUID_TICKETS


# The class for the base AST node
#
# This class will be inherited from for other types of AST nodes. Should hold all common functionality.
class BaseAstNode:

    # Initialize node with desired info
    #
    # @param children A list of child nodes
    # @param line_range A tuple of start line and end line for where this node applies
    # @param uuid A unique identifier number from a TicketCounter
    def __init__(self, uuid=None, linerange=None, **kwargs):
        self.uuid = uuid if uuid else UUID_TICKETS.get()
        self.linerange = linerange

    # Define str function to concisely summarize a node with its uuid, name/type, and relevant info
    def __str__(self):
        return '{}_{}'.format(self.uuid, type(self).__name__)

    def name(self, arg=None):
        extra = ('\\n' + str(arg)) if arg else ''
        return '"{}{}\\n{}"'.format(type(self).__name__, extra, self.uuid)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Method {}.to_3ac() is not implemented.'.format(type(self).__name__))

    # Define method for getting a GraphViz ready string
    def to_graph_viz_str(self):
        descendant_names = ', '.join([child.name() for child in self.children])
        output = '\t{} -> {{{}}};\n'.format(self.name(), descendant_names)

        for child in self.children:
            output += child.to_graph_viz_str()
        return output

    @property
    def children(self):
        return tuple([])
