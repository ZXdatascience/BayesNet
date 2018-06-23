import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, n, card):
        self.name = n
        self.card = card


class Edge:
    def __init__(self, nf, nt, t):
        """

        :param nf: Node
        :param nt: Node
        :param t: Pandas DataFrame
        """
        self.from_ = nf
        self.to = nt
        self.prob_table = t
        self.marked = False


class TabularCPD:
    def __init__(self, variable, variable_card, values, **kwargs):
        """
        :param variable: Str
        :param variable_card: int
        :param values: List[List]
        :param evidence: List
        :param evidence_card: List[int]
        """
        self.edges = []
        self.card = [variable, variable_card]
        if 'evidence' not in kwargs:
            from_node = Node('None', 0)
            to_node = Node(variable, variable_card)
            edge_prob_table = self.table(variable, variable_card, values, **kwargs)
            new_edge = Edge(from_node, to_node, edge_prob_table)
            self.edges.append(new_edge)
        else:
            evidence_card = kwargs['evidence_card']
            evidence = kwargs['evidence']
            edge_prob_table = self.table(variable, variable_card, values, **kwargs)
            for i, evid in enumerate(evidence):
                from_node = Node(evid, evidence_card[i])
                to_node = Node(variable, variable_card)
                new_edge = Edge(from_node, to_node, edge_prob_table)
                self.edges.append(new_edge)

    @staticmethod
    def table(variable, variable_card, values, **kwargs):
        if 'evidence' not in kwargs:
            transformed_values = np.array(values).T.flatten().reshape(-1, 1)
            index = np.array(range(variable_card)).reshape(variable_card,1)
            data = np.hstack((index, transformed_values))
            table = pd.DataFrame(data, columns=[variable, 'Prob'])
            table[variable] = table[variable].astype(int)
        else:
            evidence_card = kwargs['evidence_card']
            evidence = kwargs['evidence']
            transformed_values = np.array(values).T.flatten().reshape(-1,1)
            index = np.array(TabularCPD.get_index(evidence_card + [variable_card]))
            data = np.hstack((index, transformed_values))
            table = pd.DataFrame(data, columns=evidence+[variable]+['Prob'])
            table[evidence + [variable]] = table[evidence+[variable]].astype(int)
        return table

    @staticmethod
    def get_index(all_cards):
        """

        :param all_cards: List[int] {0,1}
        :return:
        """
        res = []
        largest = 1
        for card in all_cards:
            largest *= card
        for i in range(largest):
            temp = []
            num = i
            for card in all_cards[::-1]:
                temp.insert(0, num%card)
                num //= card
            res.append(temp)
        return res

    @staticmethod
    def get_index_reverse(index, cards):
        res = 0
        for i in range(len(cards)):
            res += index[i]
            res *= cards[-1-i]
        return res // cards[0]

class TableOperation:
    @staticmethod
    def multiply(table1, table2):
        common_vars = set(table1.columns).intersection(set(table2.columns))
        common_vars.remove('Prob')
        table = pd.merge(table1, table2, on=list(common_vars))
        table['Prob'] = table['Prob_x'] * table['Prob_y']
        table = table.drop(['Prob_x', 'Prob_y'], 1)
        return table

    @staticmethod
    def divide(table1, table2):
        common_vars = set(table1.columns).intersection(set(table2.columns))
        common_vars.remove('Prob')
        table = pd.merge(table1, table2, on=list(common_vars))
        table['Prob'] = table['Prob_x'] / table['Prob_y']
        table = table.drop(['Prob_x', 'Prob_y'], 1)
        return table

    @staticmethod
    def sumout(table, var):
        remain_cols = list(table.columns)
        if var not in remain_cols:
            return table
        remain_cols.remove(var)
        remain_cols.remove('Prob')
        table = table.groupby(remain_cols)['Prob'].agg('sum').reset_index()
        return table

    @staticmethod
    def select(table, evidence):
        evidence_key = evidence.keys()
        for key in evidence_key:
            if key in table.columns:
                table = table.loc[table[key] == evidence[key]]
        return table

    @staticmethod
    def normalize(table):
        prob_sum = table['Prob'].sum()
        new_table = table.copy()
        new_table['Prob'] = table['Prob']/prob_sum
        return new_table

    @staticmethod
    def has_evidence(table, evidence):
        for key in table.columns:
            if key in evidence:
                return True
        return False


class BayesianModel:
    def __init__(self, pairs):
        self.V = set()
        self.card = {}
        self.edges = []
        self.edgeFrom = {'None':[]}
        self.edgeTo = {'None':[]}
        for pair in pairs:
            self.V.add(pair[0])
            self.V.add(pair[1])
            self.edgeFrom[pair[0]] = []
            self.edgeFrom[pair[1]] = []
            self.edgeTo[pair[0]] = []
            self.edgeTo[pair[1]] = []

    def add_cpds(self, *args):
        for arg in args:
            self.card[arg.card[0]] = arg.card[1]
            for edge in arg.edges:
                self.edges.append(edge)
                self.edgeFrom[edge.from_.name].append(edge)
                self.edgeTo[edge.to.name].append(edge)


class VariableElimination:
    def __init__(self, model):
        self.model = model
        self.var_size_order = None
        node_list = list(model.V)
        node_list.sort(key=lambda x: (len(model.edgeFrom[x])) + len(model.edgeTo[x][0].prob_table.columns))
        self.var_size_order = node_list

    def query(self, targets, evidence={}):
        """

        :param targets: List
        :param evidence: Dict
        :return:
        """
        tables = {}
        for edge in self.model.edges:
            table = edge.prob_table
            table_key = list(table.columns)
            table_key.remove('Prob')
            table_key.sort()
            table_key = '#'.join(table_key)
            if table_key not in tables:
                tables[table_key] = table
        total_tables = []
        for key in tables.keys():
            total_tables.append(tables[key])
        # now we have the total tables_to_eliminate list to eliminate.
        query_res = self.eliminate(total_tables, targets, evidence)
        query_res = TableOperation.normalize(query_res)
        return query_res

    def eliminate(self, eliminate_list, targets, evidence):
        """

        :param eliminate_list:
        :param targets:
        :param evidence:
        :return:
        """
        total_targets = targets + list(evidence.keys())
        new_list = []
        for table in eliminate_list:
            if TableOperation.has_evidence(table, evidence):
                new_list.append(TableOperation.select(table, evidence))
            else:
                new_list.append(table)
        res = self.eliminate_vars(new_list, total_targets)

        return res

    def eliminate_vars(self, eliminate_list, var_s):
        """

        :param eliminate_list: Tables to eliminate
        :param var_s: The var that we want to keep
        :return:
        """
        eliminate_order = []
        var_s = set(var_s)
        for e in self.var_size_order:
            if e not in var_s:
                eliminate_order.append(e)
        for var in eliminate_order:
            eliminate_list = VariableElimination.eliminate_var(eliminate_list, var)
        eliminate_list.sort(key=lambda x: len(x.columns))
        while len(eliminate_list) >= 2:
            mul = TableOperation.multiply(eliminate_list.pop(), eliminate_list.pop())
            eliminate_list.append(mul)
        return eliminate_list[0]

    @staticmethod
    def eliminate_var(eliminate_list, var):
        tables_with_this_var = []
        new_list = []
        for e in eliminate_list:
            if var in e.columns:
                tables_with_this_var.append(e)
            else:
                new_list.append(e)
        while len(tables_with_this_var) >= 2:
            mul = TableOperation.multiply(tables_with_this_var.pop(), tables_with_this_var.pop())
            tables_with_this_var.append(mul)
        tables_with_this_var[0] = TableOperation.sumout(tables_with_this_var[0], var)
        new_list.append(tables_with_this_var[0])
        return new_list


class GibbsSampling:
    def __init__(self, model):
        """

        :param model: BayesianModel
        """
        self.model = model

    def query(self, vars,  sample_num, burn_in=0 ,evidence={}):
        samples = np.array(self.sampling(sample_num, evidence))[burn_in:]
        state_space_nodes = sorted(list(self.model.V - set(evidence.keys())))
        index = []
        for var in vars:
            index.append(state_space_nodes.index(var))
        query_samples = samples[:, index]
        # get the samples only with the variables that we query
        cards = [self.model.card[var] for var in vars]
        count_index = np.array(TabularCPD.get_index(cards))
        total_num = 1
        for card in cards:
            total_num *= card
        count = [0 for _ in range(total_num)]
        for line in query_samples:
            count[TabularCPD.get_index_reverse(line, cards)] += 1

        res_df = pd.DataFrame(count_index, columns=vars)
        res_df['Prob'] = count
        res_df = TableOperation.normalize(res_df)
        return res_df

    def draw(self, vars, values, sample_num, VE_answer, burn_in=0, evidence={}):
        samples = np.array(self.sampling(sample_num, evidence))[burn_in:]
        state_space_nodes = sorted(list(self.model.V - set(evidence.keys())))
        index = []
        for var in vars:
            index.append(state_space_nodes.index(var))
        query_samples = samples[:, index]
        # get the samples only with the variables that we query
        probs = []
        count = 0
        compare = np.sum(query_samples == values, axis=1)
        for i, comp in enumerate(compare):
            if comp == len(vars):
                count += 1
            probs.append(count / (i + 1))
        plt.plot(probs, color='blue')
        plt.axhline(y=VE_answer,color='red',linestyle='dashed')
        plt.show()

    def sampling(self, sample_num, evidence={}):
        state_dp = {}
        state_space_nodes = sorted(list(self.model.V - set(evidence.keys())))
        new_evidence = dict(zip(state_space_nodes, [0 for _ in range(len(state_space_nodes))]))
        samples = [[0 for _ in range(len(state_space_nodes))]]
        # step 1: full assignment
        all_evidence = {**evidence, **new_evidence}
        for i in range(sample_num):
            # step 2: sample all the variables in order one by one with other variable fixed.
            sample = []
            for var in state_space_nodes:
                all_evidence.pop(var)
                evidence_list = [[pair[0], str(pair[1])] for pair in sorted(all_evidence.items())]
                current_state = ''.join([''.join(pair) for pair in evidence_list])
                # here I used dp to store all the transition probability each state could have.
                # This is not practical if there thousands of variables.
                if current_state in state_dp:
                    distribution = state_dp[current_state]
                else:
                    related_tables, related_vars = self.get_tables_evi(var)
                    related_vars.remove(var)
                    related_evidence = dict((k, all_evidence[k]) for k in related_vars)
                    distribution_table = GibbsSampling.eliminate(related_tables, related_evidence)
                    distribution = distribution_table['Prob'].values
                    state_dp[current_state] = distribution
                next_state = np.random.choice(range(len(distribution)), 1, p=distribution)[0]
                all_evidence[var] = next_state
                sample.append(next_state)
            samples.append(sample)
        return samples


    @staticmethod
    def eliminate(related_tables, related_evidence):
        selected = []
        for table in related_tables:
            selected.append(TableOperation.select(table, related_evidence))
        res = selected[0]
        for table in selected[1:]:
            res = TableOperation.multiply(res, table)
        res = TableOperation.normalize(res)
        return res

    def get_tables_evi(self, var):
        tables = {}
        related_nodes = set()
        var_edges = self.model.edgeFrom[var] + self.model.edgeTo[var]
        for edge in var_edges:
            table = edge.prob_table
            table_key = list(table.columns)
            table_key.remove('Prob')
            for key in table_key:
                related_nodes.add(key)
            table_key.sort()
            table_key = '#'.join(table_key)
            if table_key not in tables:
                tables[table_key] = table
        total_tables = []
        for key in tables.keys():
            total_tables.append(tables[key])

        return total_tables, related_nodes



if __name__ == '__main__':
    model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])
    # Defining individual CPDs.
    cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
    cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])

    # The representation of CPD in pgmpy is a bit different than the CPD shown in the above picture. In pgmpy the colums
    # are the evidences and rows are the states of the variable. So the grade CPD is represented like this:
    #
    #    +---------+---------+---------+---------+---------+
    #    | diff    | intel_0 | intel_0 | intel_1 | intel_1 |
    #    +---------+---------+---------+---------+---------+
    #    | intel   | diff_0  | diff_1  | diff_0  | diff_1  |
    #    +---------+---------+---------+---------+---------+
    #    | grade_0 | 0.3     | 0.05    | 0.9     | 0.5     |
    #    +---------+---------+---------+---------+---------+
    #    | grade_1 | 0.4     | 0.25    | 0.08    | 0.3     |
    #    +---------+---------+---------+---------+---------+
    #    | grade_2 | 0.3     | 0.7     | 0.02    | 0.2     |
    #    +---------+---------+---------+---------+---------+
    #
    cpd_g = TabularCPD(variable='G', variable_card=3,
                       values=[[0.3, 0.05, 0.9, 0.5],
                               [0.4, 0.25, 0.08, 0.3],
                               [0.3, 0.7, 0.02, 0.2]],
                       evidence=['I', 'D'],
                       evidence_card=[2, 2])

    cpd_l = TabularCPD(variable='L', variable_card=2,
                       values=[[0.1, 0.4, 0.99],
                               [0.9, 0.6, 0.01]],
                       evidence=['G'],
                       evidence_card=[3])

    cpd_s = TabularCPD(variable='S', variable_card=2,
                       values=[[0.95, 0.2],
                               [0.05, 0.8]],
                       evidence=['I'],
                       evidence_card=[2])

    # Associating the CPDs with the network
    model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
    infer_gibbs = GibbsSampling(model)
    infer_ve = VariableElimination(model)
    print(infer_ve.query(['G', 'L'], evidence={'D': 0, 'I': 1}))
    print(infer_gibbs.query(['G', 'L'], 200000, evidence={'D': 0, 'I': 1}))
