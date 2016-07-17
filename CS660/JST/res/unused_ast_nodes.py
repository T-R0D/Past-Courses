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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

from ast.base_ast_node import BaseAstNode


##
# REVISIT ME - May need to add in stuff for line numbers for jumps in 3ac
##
class Break(BaseAstNode):
    def __init__(self, **kwargs):
        super(Break, self).__init__(**kwargs)

    @property
    def children(self):
        children = []
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class Case(BaseAstNode):
    def __init__(self, expression, statement_list=None, **kwargs):
        super(Case, self).__init__(**kwargs)

        self.expression = expression
        self.statement_list = statement_list if statement_list else []

    @property
    def children(self):
        children = []
        children.append(self.expression)
        children.extend(self.statement_list)
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class Continue(BaseAstNode):
    def __init__(self, **kwargs):
        super(Continue, self).__init__(**kwargs)

    @property
    def children(self):
        children = []
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class Default(BaseAstNode):
    def __init__(self, statement_list=None, **kwargs):
        super(Default, self).__init__(**kwargs)

        self.statement_list = statement_list if statement_list else []

    @property
    def children(self):
        children = []
        children.extend(self.statement_list)
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Default AST node.')


class Goto(BaseAstNode):
    def __init__(self, name, **kwargs):
        super(Goto, self).__init__(**kwargs)

        self.name = name

    @property
    def children(self):
        children = []
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


# TODO (Shubham) What would this be used for? Possibly removable
class IdentifierType(BaseAstNode):
    def __init__(self, names, **kwargs):
        super(IdentifierType, self).__init__(**kwargs)

        self.names = names

    @property
    def children(self):
        children = []
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class Switch(BaseAstNode):
    def __init__(self, conditional, body_statement, **kwargs):
        super(Switch, self).__init__(**kwargs)

        self.conditional = conditional
        self.body_statement = body_statement

    @property
    def children(self):
        children = []
        children.append(self.conditional)
        children.append(self.body_statement)
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))


class TernaryOperator(BaseAstNode):
    def __init__(self, conditional, if_true_expression, if_false_expression, **kwargs):
        super(TernaryOperator, self).__init__(**kwargs)

        self.conditional = conditional
        self.if_true_expression = if_true_expression
        self.if_false_expression = if_false_expression

    @property
    def children(self):
        children = []
        children.append(self.conditional)
        children.append(self.if_true_expression)
        children.append(self.if_false_expression)
        return tuple(children)

    def to_3ac(self, include_source=False):
        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'.format(type(self).__name__))

