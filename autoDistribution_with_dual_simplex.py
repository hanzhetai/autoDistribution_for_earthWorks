import numpy as np
import math

# 显示设置
np.set_printoptions(linewidth=10000)


class inputTransform:
    def __init__(self):
        # L_array, length of routes between sections
        self.path_L = r'D:\testSample_autoDistribution\L_array.csv'
        # L_exterior_array, length of routes from outside earthworkds sources to each section
        self.path_L_exterior = r'D:\testSample_autoDistribution\L_exterior_array.csv'
        # H_exterior_array, human factors, represents the import willingness for each section, set 0 if you wanted to probhit the specific section from importing earthworks from outside
        self.path_H_exterior = r'D:\testSample_autoDistribution\H_exterior_array.csv'
        # H_array, human factors, represents the willingness for sections to interchange resources, set 0 if you wanted to probhit sections from exporting eathworks to the specific one
        self.path_H = r'D:\testSample_autoDistribution\H_array.csv'
        # phi_array, represents the fill/cut factor in paved area
        self.path_phi = r'D:\testSample_autoDistribution\phi_array.csv'
        # theta_array, represents the fill/cut factor in unpaved area
        self.path_theta = r'D:\testSample_autoDistribution\theta_array.csv'
        # F_array, represents the embankment quantity of earthworks in paved area for each section
        self.path_F = r'D:\testSample_autoDistribution\F_array.csv'
        # D_array, represents the embankment quantity of earthworks in unpaved area for each section
        self.path_D = r'D:\testSample_autoDistribution\D_array.csv'
        # N_array, represents the excavation quantity of earthworks for each section
        self.path_N = r'D:\testSample_autoDistribution\N_array.csv'
        # prior_distribute_array, represents the priority of sections to distribute earthworks, lower value means higher priority
        self.path_prior_distribute_array = r'D:\testSample_autoDistribution\prior_distribute_array.csv'
        # prior_receive_array, represents the priority of sections to receive earthworks, lower value means higher priority
        self.path_prior_receive_array = r'D:\testSample_autoDistribution\prior_distribute_array.csv'
        # theta_0 factor, represents the fill/cut factor in paved area when using exterior earthworks
        self.phi_0 = 0.9
        # theta_0 factor, represents the fill/cut factor in unpaved area when using exterior earthworks
        self.theta_0 = 0.9
        # indexes in the list represent the indexes of the base treatment section
        self.earth_treatment_index = [4, 11, 18, 25, 32, 39, 46]
        # put index here if you want to prohibit the section from receiving exterior earthworks
        self.prohibited_section = []
        # set the default earthworks source quantity, will fix to an auto default value by sum the F and N
        self.exterior_import = 10000000

        # 加载距离权重L矩阵 (L_array, length of routes between sections)
        with open(self.path_L, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.L_matrix = tmp[0:].astype(np.float64)

        # 加载距离权重L_exterior矩阵 (L_exterior_array, length of routes from outside earthworkds sources to each section)
        with open(self.path_L_exterior, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.L_exterior_list = tmp[0:].astype(np.float64)

        # 加载人为因素H矩阵 (H_array, human factors, represents the willingness for sections to interchange resources, set 0 if you wanted to probhit sections from exporting eathworks to the specific one)
        with open(self.path_H, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.H_matrix = tmp[0:].astype(np.float64)

        # 加载人为因素H_exterior矩阵 (H_exterior_array, human factors, represents the import willingness for each section, set 0 if you wanted to probhit the specific section from importing earthworks from outside)
        with open(self.path_H_exterior, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.H_exterior_matrix = tmp[0:].astype(np.float64)

        # 加载土基区填挖比phi矩阵 (phi_array, represents the fill/cut factor in paved area)
        with open(self.path_phi, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.phi_matrix = tmp[0:].astype(np.float64)

        # 加载土面区填挖比theta矩阵 (theta_array, represents the fill/cut factor in unpaved area)
        with open(self.path_theta, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.theta_matrix = tmp[0:].astype(np.float64)

        # 加载土基区填方F矩阵 (F_array, represents the embankment quantity of earthworks in paved area for each section)
        with open(self.path_F, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.F_matrix = tmp[0:].astype(np.float64)

        # 加载土面区填方D矩阵 (D_array, represents the embankment quantity of earthworks in unpaved area for each section)
        with open(self.path_D, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.D_matrix = tmp[0:].astype(np.float64)

        # 加载挖方N矩阵 (N_array, represents the excavation quantity of earthworks for each section)
        with open(self.path_N, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.N_matrix = tmp[0:].astype(np.float64)

        # 加载分配优先矩阵 (prior_distribute_array)
        with open(self.path_prior_distribute_array, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.prior_distribute_array = tmp[0:].astype(np.float64)

        # 加载分配优先矩阵 (prior_receive_array)
        with open(self.path_prior_receive_array, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.prior_receive_array = tmp[0:].astype(np.float64)

        # 依照L的长度判断分区数量
        self.sec_num = len(self.L_matrix.tolist())

        # 生成求解矩阵(1表示存在待求变量，初始矩阵均为代求解)
        self.initial_P_matrix = np.ones(self.L_matrix.shape)
        self.initial_Q_matrix = np.ones(self.L_matrix.shape)
        self.initial_Epsilon_for_pavedArea_matrix = np.ones(self.F_matrix.shape)
        self.initial_Epsilon_for_unPavedArea_matrix = np.ones(self.F_matrix.shape)

        # 判断借方区及填方区
        # get self ratio for embankment_area
        phi_diagnoal = self.phi_matrix.diagonal()
        theta_diagnoal = self.theta_matrix.diagonal()

        paved_embankment = np.multiply(self.F_matrix, phi_diagnoal)
        unpaved_embankment = np.multiply(self.D_matrix, theta_diagnoal)
        self.total_embankment = paved_embankment + unpaved_embankment

        if self.earth_treatment_index:
            for section_Index in self.earth_treatment_index:
                # output from base treatment area should not be used in paved area
                self.initial_P_matrix[section_Index, :] = 0
                # output from base treatment area used in unpaved area should not be used by itself
                for section_col_index in self.earth_treatment_index:
                    self.initial_Q_matrix[section_Index, section_col_index] = 0

    def P_coefficient_matrix(self):
        distribution_by_exterior_resource = np.array([self.initial_Epsilon_for_pavedArea_matrix * self.phi_0],
                                                     dtype=float)
        restricted_by_H = self.initial_P_matrix * self.H_matrix
        combined_array = np.vstack((restricted_by_H * self.phi_matrix, distribution_by_exterior_resource))
        return combined_array

    def Q_coefficient_matrix(self):
        distribution_by_exterior_resource = np.array(self.initial_Epsilon_for_unPavedArea_matrix * self.theta_0,
                                                     dtype=float)
        restricted_by_H = self.initial_Q_matrix * self.H_matrix
        combined_array = np.vstack((restricted_by_H * self.theta_matrix, distribution_by_exterior_resource))
        return combined_array

    def H_matrix_refit(self):
        H_matrix_refit = np.copy(self.H_matrix)
        update_row = np.ones((1, H_matrix_refit.shape[1]))
        if self.prohibited_section:
            for sec in self.prohibited_section:
                update_row[0, int(sec)] = 0
            return np.vstack((H_matrix_refit, update_row))
        else:
            return np.vstack((H_matrix_refit, update_row))

    def L_matrix_refit(self):
        L_matrix_refit = np.copy(self.L_matrix)
        L_exterior_matrix = np.copy(np.array(self.L_exterior_list))
        distance_matrix = np.vstack((L_matrix_refit, L_exterior_matrix))
        return distance_matrix

    def zerofied_non_focused_row(self, array, selected_row):
        array[0: selected_row, :] = 0
        array[selected_row + 1:, :] = 0
        array_flat = [elem for row in array for elem in row]
        return array_flat

    def zerofied_non_focused_column(self, array, selected_column):
        array[:, 0: selected_column] = 0
        array[:, selected_column + 1:] = 0
        array_flat = [elem for row in array for elem in row]
        return array_flat

    def embankment_constraints_coefficients_on_pavedArea(self):
        embankment_constraints_coefficients_on_pavedArea = np.copy(self.P_coefficient_matrix())
        len_of_instance = len(embankment_constraints_coefficients_on_pavedArea[0])

        embankment_constraints_coefficients_on_unPavedArea_with_Zeros = np.zeros(
            embankment_constraints_coefficients_on_pavedArea.shape)
        embankment_constraints_coefficients_on_pavedArea = np.hstack((embankment_constraints_coefficients_on_pavedArea,
                                                                      embankment_constraints_coefficients_on_unPavedArea_with_Zeros))

        costraints_coefficients_list = []
        for i in range(0, len_of_instance):
            coefficients_matrix = np.copy(embankment_constraints_coefficients_on_pavedArea)
            focused = inputTransform.zerofied_non_focused_column(self, coefficients_matrix, i)
            costraints_coefficients_list.append(focused)
        return np.array(costraints_coefficients_list, dtype=float)

    def embankment_constraints_coefficients_on_unPavedArea(self):
        embankment_constraints_coefficients_on_unPavedArea = np.copy(self.Q_coefficient_matrix())
        init_of_instance = len(embankment_constraints_coefficients_on_unPavedArea[0])

        embankment_constraints_coefficients_on_pavedArea_with_Zeros = np.zeros(
            embankment_constraints_coefficients_on_unPavedArea.shape)
        embankment_constraints_coefficients_on_unPavedArea = np.hstack((
            embankment_constraints_coefficients_on_pavedArea_with_Zeros,
            embankment_constraints_coefficients_on_unPavedArea))
        end_of_instance = len(embankment_constraints_coefficients_on_unPavedArea[0])

        costraints_coefficients_list = []
        for i in range(init_of_instance, end_of_instance):
            coefficients_matrix = np.copy(embankment_constraints_coefficients_on_unPavedArea)
            focused = inputTransform.zerofied_non_focused_column(self, coefficients_matrix, i)
            costraints_coefficients_list.append(focused)
        return np.array(costraints_coefficients_list, dtype=float)

    def excavation_constraints_coefficients(self):
        excavation_constraints_coefficients_on_pavedArea = np.copy(self.P_coefficient_matrix())
        excavation_constraints_coefficients_on_pavedArea[excavation_constraints_coefficients_on_pavedArea != 0] = -1

        excavation_constraints_coefficients_on_unPavedArea = np.copy(self.Q_coefficient_matrix())
        excavation_constraints_coefficients_on_unPavedArea[excavation_constraints_coefficients_on_unPavedArea != 0] = -1

        combine_excavation_constraints = np.hstack(
            (excavation_constraints_coefficients_on_pavedArea, excavation_constraints_coefficients_on_unPavedArea))

        costraints_coefficients_list = []
        len_of_instance = len(combine_excavation_constraints)

        for i in range(0, len_of_instance):
            coefficients_matrix = np.copy(combine_excavation_constraints)
            focused = inputTransform.zerofied_non_focused_row(self, coefficients_matrix, i)
            costraints_coefficients_list.append(focused)
        return np.array(costraints_coefficients_list, dtype=float)

    # coefficients matrix A
    def combine_constraints_coefficients(self):
        combine_constraints_coefficients = np.vstack(
            (inputTransform.embankment_constraints_coefficients_on_pavedArea(self), \
             inputTransform.embankment_constraints_coefficients_on_unPavedArea(self), \
             inputTransform.excavation_constraints_coefficients(self)
             )
        )
        return combine_constraints_coefficients

    def embankment_constraints_on_pavedArea(self):
        F_array = np.copy(self.F_matrix)
        return F_array

    def embankment_constraints_on_unPavedArea(self):
        D_array = np.copy(self.D_matrix)
        return D_array

    def excavation_constraints(self):
        N_array = np.copy(self.N_matrix)

        # Create a new ndarray with the desired size (original size + 1)
        N_array_extend = np.empty(N_array.shape[0] + 1)

        # Copy the elements from the original ndarray to the new ndarray
        N_array_extend[:-1] = -N_array

        # Add the new element to the last position in the new ndarray
        N_array_extend[-1] = -self.exterior_import
        return N_array_extend

    # coefficients array b
    def combine_constraints(self):
        combine_constraints = np.hstack(
            (inputTransform.embankment_constraints_on_pavedArea(self), \
             inputTransform.embankment_constraints_on_unPavedArea(self), \
             inputTransform.excavation_constraints(self)
             ))
        return combine_constraints

    # constraints array c
    def combine_objective(self):
        L_matrix = np.copy(self.L_matrix_refit())
        H_matrix = np.copy(self.H_matrix_refit())

        prior_distribute_array = np.copy(self.prior_distribute_array)
        update_distribute_row = np.ones((1, prior_distribute_array.shape[1]))

        prior_distribute_array_expand = np.vstack((prior_distribute_array, update_distribute_row))

        prior_receive_array = np.copy(self.prior_receive_array)
        update_receive_row = np.ones((1, prior_receive_array.shape[1]))
        prior_receive_array_expand = np.vstack((prior_receive_array, update_receive_row))

        objective = L_matrix[:] * H_matrix[:] * prior_distribute_array_expand[:] * prior_receive_array_expand[:]
        combine_objective = np.hstack((objective, objective))
        constraints_flat = [elem for row in combine_objective for elem in row]
        return np.array(constraints_flat, dtype=float)


def dual_simplex(c, A, b):
    # Transform the problem to its dual
    c_dual = c
    A_dual = A.T
    b_dual = -b

    # Form the tableau
    tableau = np.column_stack((A_dual, np.eye(A_dual.shape[0]), c_dual.reshape(-1, 1)))
    tableau = np.vstack((tableau, np.append(b_dual, np.zeros(A_dual.shape[0] + 1))))
    tableau[-1, -1] = 0

    print(f'problem transposed\n{tableau}')

    iter = 1
    while np.any(tableau[-1, :-1] < 0):
        print(f'===========================start iteration {iter}===========================')
        # Bland's rule for entering variable: Choose the first variable with a negative reduced cost
        pivot_col = np.where(tableau[-1, :-1] < 0)[0][0]

        # #############for display only !!!! potentially decrease efficiency when activated !!!
        # # Highlight Column
        # print(f'HIGHTLIGHT_PIVOT_COLUMN {pivot_col}')
        # highlight_col = pivot_col
        # for row_idx, row in enumerate(tableau):
        #     col_str = []
        #     for col_idx, elem in enumerate(row):
        #         if col_idx == highlight_col:
        #             col_str.append(f"\033[1;31m{round(elem, 3):15}\033[0m")  # Red highlight for row, 5 spaces
        #         else:
        #             col_str.append(
        #                 f"\033[1;36m{round(elem, 3):15}\033[0m")  # Salmon hightlight for the rest of row and 5 spaces for other elements
        #     print("[" + "".join(col_str) + "]")
        # #############

        valid_ratios = np.array([(i, tableau[i, -1] / tableau[i, pivot_col])
                                 for i in range(tableau.shape[0] - 1)
                                 if tableau[i, pivot_col] > 0])

        if valid_ratios.size == 0:
            raise ValueError("Problem is unbounded.")

        pivot_row = valid_ratios[np.argmin(valid_ratios[:, 1]), 0].astype(int)

        # #############for display only !!!! potentially decrease efficiency when activated !!!
        # # Highlight Row
        # print(f'HIGHTLIGHT_PIVOT_ROW {pivot_row}')
        # highlight_row = pivot_row
        # for row_idx, row in enumerate(tableau):
        #     row_str = []
        #     for col_idx, elem in enumerate(row):
        #         if row_idx == highlight_row:
        #             row_str.append(f"\033[1;31m{round(elem, 3):15}\033[0m")  # Red highlight for row, 5 spaces
        #         else:
        #             row_str.append(
        #                 f"\033[1;36m{round(elem, 3):15}\033[0m")  # Salmon hightlight for the rest of row and 5 spaces for other elements
        #     print("[" + "".join(row_str) + "]")
        # #############

        # #############for display only !!!! potentially decrease efficiency when activated !!!
        # # Highlight Element
        # print(f'HIGHTLIGHT_PIVOT_ELEMENT {tableau[pivot_row, pivot_col]}')
        # highlight_element = (pivot_row, pivot_col)
        # for row_idx, row in enumerate(tableau):
        #     row_str = []
        #     for col_idx, elem in enumerate(row):
        #         if (row_idx, col_idx) == highlight_element:
        #             row_str.append(f"\033[1;31m{round(elem, 3):15}\033[0m")  # Red highlight for row, 5 spaces
        #         else:
        #             row_str.append(
        #                 f"\033[1;36m{round(elem, 3):15}\033[0m")  # Salmon hightlight for the rest of row and 5 spaces for other elements
        #     print("[" + "".join(row_str) + "]")
        # #############

        # Check if the pivot element is non-zero (or not very close to zero)
        if abs(tableau[pivot_row, pivot_col]) < 1e-10:
            raise ValueError("Pivot element is too close to zero, numerical instability may occur.")

        tableau[pivot_row] /= tableau[pivot_row, pivot_col]

        # #############for display only !!!! potentially decrease efficiency when activated !!!
        # # Highlight Row
        # print(f'HIGHTLIGHT_REFITED_PIVOT_ROW {pivot_row}')
        # highlight_row = pivot_row
        # for row_idx, row in enumerate(tableau):
        #     row_str = []
        #     for col_idx, elem in enumerate(row):
        #         if row_idx == highlight_row:
        #             row_str.append(f"\033[1;31m{round(elem, 3):15}\033[0m")  # Red highlight for row, 5 spaces
        #         else:
        #             row_str.append(
        #                 f"\033[1;36m{round(elem, 3):15}\033[0m")  # Salmon hightlight for the rest of row and 5 spaces for other elements
        #     print("[" + "".join(row_str) + "]")
        # #############

        for i in range(tableau.shape[0]):
            if (i != pivot_row and tableau[i, pivot_col] != 0):
                print(f'refit row {i} ====> ')

                # #############for display only !!!! potentially decrease efficiency when activated !!!
                # # Highlight Row
                # print(f'HIGHTLIGHT_ROW {i} _TO_FIT_PIVOT_ROW {pivot_row}')
                # highlight_row = pivot_row
                # for row_idx, row in enumerate(tableau):
                #     row_str = []
                #     for col_idx, elem in enumerate(row):
                #         if row_idx == highlight_row:
                #             row_str.append(
                #                 f"\033[1;31m{round(elem, 3):15}\033[0m")  # Red highlight for row pivot, 5 spaces
                #         elif row_idx == i:
                #             row_str.append(
                #                 f"\033[1;33m{round(elem, 3):15}\033[0m")  # Yellow highlight for row to be modified, 5 spaces
                #         else:
                #             row_str.append(
                #                 f"\033[1;36m{round(elem, 3):15}\033[0m")  # Salmon hightlight for the rest of row and 5 spaces for other elements
                #     print("[" + "".join(row_str) + "]")
                # #############

                tableau[i] -= tableau[pivot_row] * tableau[i, pivot_col]

                # ############for display only !!!! potentially decrease efficiency when activated !!!
                # # Highlight Row
                # print(f'HIGHTLIGHT_ROW {i} _TO_FIT_PIVOT_ROW {pivot_row}')
                # highlight_row = pivot_row
                # for row_idx, row in enumerate(tableau):
                #     row_str = []
                #     for col_idx, elem in enumerate(row):
                #         if row_idx == highlight_row:
                #             row_str.append(
                #                 f"\033[1;31m{round(elem, 3):15}\033[0m")  # Red highlight for row pivot, 5 spaces
                #         elif row_idx == i:
                #             row_str.append(
                #                 f"\033[1;34m{round(elem, 3):15}\033[0m")  # Blue highlight for row already modified, 5 spaces
                #         else:
                #             row_str.append(
                #                 f"\033[1;36m{round(elem, 3):15}\033[0m")  # Salmon hightlight for the rest of row and 5 spaces for other elements
                #     print("[" + "".join(row_str) + "]")
                # #############

        iter += 1
    # print(f'final result\n{tableau}')
    solution_row = tableau[-1]
    optimal_value = solution_row[-1]
    var_index_init = len(A_dual[0])
    # default set slack varibales connected to excavation inequality constraints and exterior input inequality constraints
    slack_num = int((var_index_init - 1) / 3) + 1
    print('var_index_init', var_index_init)
    print('slack num', slack_num)
    solution = solution_row[var_index_init: -slack_num - 1]
    return optimal_value, solution


# create instance
my_instance = inputTransform()

# get_num_of_create instance
num_of_instance = len(my_instance.P_coefficient_matrix())

# load data from instance
c = my_instance.combine_objective()
A = my_instance.combine_constraints_coefficients()

# introduce slack_variables
slack_variables_matrix = np.eye(A.shape[0])

for i in range(len(A)):
    if (i >= (num_of_instance - 1) * 2):
        slack_variables_matrix[i] = slack_variables_matrix[i] * (-1)
slack_variables_matrix = slack_variables_matrix[:, (num_of_instance - 1) * 2:]

# expand A matrix
A = np.hstack((A, slack_variables_matrix))

# introduce slack variables coefficients of zeros
slack_variables_coefficients = np.zeros((len(slack_variables_matrix[0]),))

# expand c array
c = np.hstack((c, slack_variables_coefficients))

b = my_instance.combine_constraints()

# output
optimal_value, solution = dual_simplex(c, A, b)
print("Optimal value:", optimal_value)

solution_reshape = solution.reshape(num_of_instance, (num_of_instance - 1) * 2)
print(f"Solution: \n {solution_reshape}")
np.savetxt(r"D:\testSample_autoDistribution\solution.csv", solution_reshape, delimiter=",")

paved_solution = solution_reshape[:, :num_of_instance - 1]
print(f"Paved_Solution: \n {paved_solution}")
np.savetxt(r"D:\testSample_autoDistribution\paved_solution.csv", paved_solution, delimiter=",")

unpaved_solution = solution_reshape[:, num_of_instance - 1:]
print(f"unPaved_Solution: \n {unpaved_solution}")
np.savetxt(r"D:\testSample_autoDistribution\unpaved_solution.csv", unpaved_solution, delimiter=",")
