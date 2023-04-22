import numpy as np
import math

#显示设置
np.set_printoptions(linewidth=1000)


class inputTransform:
    def __init__(self):
        # 输入常量
        # 距离权重L矩阵
        self.path_L = r'D:\testSample_autoDistribution\L_array.csv'
        # 距离权重L_exterior矩阵
        self.path_L_exterior = r'D:\testSample_autoDistribution\L_exterior_array.csv'
        # 人为因素H_exterior矩阵
        self.path_H_exterior = r'D:\testSample_autoDistribution\H_exterior_array.csv'
        # 人为因素H矩阵
        self.path_H = r'D:\testSample_autoDistribution\H_array.csv'
        # 土基区填挖比phi矩阵
        self.path_phi = r'D:\testSample_autoDistribution\phi_array.csv'
        # 土面区填挖比theta矩阵
        self.path_theta = r'D:\testSample_autoDistribution\theta_array.csv'
        # 土基区填方F矩阵
        self.path_F = r'D:\testSample_autoDistribution\F_array.csv'
        # 土面区填方D矩阵
        self.path_D = r'D:\testSample_autoDistribution\D_array.csv'
        # 挖方N矩阵
        self.path_N = r'D:\testSample_autoDistribution\N_array.csv'
        # 外借土方土基区填挖比phi_0参数
        self.phi_0 = 0.85
        # 外借土方土面区填挖比theta_0参数
        self.theta_0 = 0.88
        # 地基处理子项位于第几行
        self.earth_treatment_index = 5
        # 不得外借土方的标段
        self.prohibited_section = []
        # 设置借方(尽可能设置大值)
        self.exterior_inport = 900

        with open(self.path_L, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.L_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载距离权重L_exterior矩阵
        with open(self.path_L_exterior, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.L_exterior_list = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载人为因素H矩阵
        with open(self.path_H, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.H_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载人为因素H_exterior矩阵
        with open(self.path_H_exterior, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.H_exterior_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载土基区填挖比phi矩阵
        with open(self.path_phi, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.phi_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载土面区填挖比theta矩阵
        with open(self.path_theta, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.theta_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载土基区填方F矩阵
        with open(self.path_F, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.F_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载土面区填方D矩阵
        with open(self.path_D, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.D_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 加载挖方N矩阵
        with open(self.path_N, encoding='utf-8-sig') as f:
            tmp = np.loadtxt(f, str, delimiter=",")
            self.N_matrix = tmp[0:].astype(np.float64)  # 加载数据部

        # 依照L的长度判断分区数量
        self.sec_num = len(self.L_matrix.tolist())

        # 生成求解矩阵(1表示存在待求变量，初始矩阵均为代求解)
        self.initial_P_matrix = np.ones(self.L_matrix.shape)
        self.initial_Q_matrix = np.ones(self.L_matrix.shape)
        self.initial_Epsilon_for_pavedArea_matrix = np.ones(self.F_matrix.shape)
        self.initial_Epsilon_for_unPavedArea_matrix = np.ones(self.F_matrix.shape)

        # 判断借方区及填方区
        # get diagonal ratio for embankment_area
        phi_diagnoal = self.phi_matrix.diagonal()
        theta_diagnoal = self.theta_matrix.diagonal()

        paved_embankment = np.multiply(self.F_matrix, phi_diagnoal)
        unpaved_embankment = np.multiply(self.D_matrix, theta_diagnoal)
        self.total_embankment = paved_embankment + unpaved_embankment

        # 变量约减(尚未考虑地基处理区的约减与修正，后续补充)
        for i in range(len(self.total_embankment)):
            if self.total_embankment[i] < self.N_matrix[i]:
                # 余方区不需要外借土源，对应的值归0
                self.initial_Epsilon_for_pavedArea_matrix[i] = 0
                self.initial_Epsilon_for_unPavedArea_matrix[i] = 0
            else:
                # 其余的区域为缺方区，优先供应本区的土方，因此其余列的值归0
                self.initial_P_matrix[i, :] = 0
                self.initial_Q_matrix[i, :] = 0

                # 恢复缺方区的求解值(为简化代码的中间过程)
                self.initial_P_matrix[i, i] = 1
                self.initial_Q_matrix[i, i] = 1
                # 对于缺方区，优先满足本区土基区的土方
                # 判断本区挖方是否恰好满足本区填方
                if self.total_embankment[i] == self.N_matrix[i]:
                    # 对应的epsilon归零
                    self.initial_Epsilon_for_pavedArea_matrix[i] = 0
                    self.initial_Epsilon_for_unPavedArea_matrix[i] = 0

    def P_coefficient_matrix(self):
        distribution_by_exterior_resource = np.array([self.initial_Epsilon_for_pavedArea_matrix * self.phi_0],
                                                     dtype=float)
        combined_array = np.vstack((self.initial_P_matrix * self.phi_matrix, distribution_by_exterior_resource))
        # 生成一列数值均为0的array，代表外借土源的系数
        extend_column = np.zeros((combined_array.shape[0], 1))
        # 将该array添加于matrix的右侧，代表各区向该区域调配的土方，由于该列代表外借土源，因此默认调往该区的土方为0
        combined_array = np.concatenate((combined_array, extend_column), axis=1)
        len_of_instance = len(combined_array[0])
        combined_array = combined_array[:, :len_of_instance - 1]
        return combined_array

    def Q_coefficient_matrix(self):
        distribution_by_exterior_resource = np.array(self.initial_Epsilon_for_unPavedArea_matrix * self.theta_0,
                                                     dtype=float)
        combined_array = np.vstack((self.initial_Q_matrix * self.theta_matrix, distribution_by_exterior_resource))
        # 生成一列数值均为0的array，代表外借土源的系数
        extend_column = np.zeros((combined_array.shape[0], 1))
        # 将该array添加于matrix的右侧，代表各区向该区域调配的土方，由于该列代表外借土源，因此默认调往该区的土方为0
        combined_array = np.concatenate((combined_array, extend_column), axis=1)
        len_of_instance = len(combined_array[0])
        combined_array = combined_array[:, :len_of_instance - 1]
        return combined_array

    def H_matrix_refit(self):
        H_matrix_refit = np.copy(self.H_matrix)
        update_row = np.ones((1, H_matrix_refit.shape[1]))
        update_row[0, -1] = 0
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

    # Refit II
    def embankment_constraints_coefficients_on_pavedArea(self):
        embankment_constraints_coefficients_on_pavedArea = np.copy(self.P_coefficient_matrix())
        len_of_instance = len(embankment_constraints_coefficients_on_pavedArea[0])

        embankment_constraints_coefficients_on_unPavedArea_with_Zeros = np.zeros(
            embankment_constraints_coefficients_on_pavedArea.shape)
        embankment_constraints_coefficients_on_pavedArea = np.hstack((embankment_constraints_coefficients_on_pavedArea,
                                                                      embankment_constraints_coefficients_on_unPavedArea_with_Zeros))

        print(embankment_constraints_coefficients_on_pavedArea)
        costraints_coefficients_list = []
        for i in range(0, len_of_instance):
            coefficients_matrix = np.copy(embankment_constraints_coefficients_on_pavedArea)
            focused = inputTransform.zerofied_non_focused_column(self, coefficients_matrix, i)
            costraints_coefficients_list.append(focused)
        return np.array(costraints_coefficients_list, dtype=float)

    # Refit II
    def embankment_constraints_coefficients_on_unPavedArea(self):
        embankment_constraints_coefficients_on_unPavedArea = np.copy(self.Q_coefficient_matrix())
        init_of_instance = len(embankment_constraints_coefficients_on_unPavedArea[0])

        embankment_constraints_coefficients_on_pavedArea_with_Zeros = np.zeros(
            embankment_constraints_coefficients_on_unPavedArea.shape)
        embankment_constraints_coefficients_on_unPavedArea = np.hstack((
                                                                       embankment_constraints_coefficients_on_pavedArea_with_Zeros,
                                                                       embankment_constraints_coefficients_on_unPavedArea))
        end_of_instance = len(embankment_constraints_coefficients_on_unPavedArea[0])

        print(embankment_constraints_coefficients_on_unPavedArea)
        costraints_coefficients_list = []
        for i in range(init_of_instance, end_of_instance):
            coefficients_matrix = np.copy(embankment_constraints_coefficients_on_unPavedArea)
            focused = inputTransform.zerofied_non_focused_column(self, coefficients_matrix, i)
            costraints_coefficients_list.append(focused)
        return np.array(costraints_coefficients_list, dtype=float)

    # Refit II
    def excavation_constraints_coefficients(self):
        excavation_constraints_coefficients_on_pavedArea = np.copy(self.P_coefficient_matrix())
        excavation_constraints_coefficients_on_pavedArea[excavation_constraints_coefficients_on_pavedArea != 0] = -1

        excavation_constraints_coefficients_on_unPavedArea = np.copy(self.Q_coefficient_matrix())
        excavation_constraints_coefficients_on_unPavedArea[excavation_constraints_coefficients_on_unPavedArea != 0] = -1

        combine_excavation_constraints = np.hstack(
            (excavation_constraints_coefficients_on_pavedArea, excavation_constraints_coefficients_on_unPavedArea))
        print(combine_excavation_constraints)

        costraints_coefficients_list = []
        len_of_instance = len(combine_excavation_constraints)

        for i in range(0, len_of_instance):
            coefficients_matrix = np.copy(combine_excavation_constraints)
            focused = inputTransform.zerofied_non_focused_row(self, coefficients_matrix, i)
            costraints_coefficients_list.append(focused)
        return np.array(costraints_coefficients_list, dtype=float)

    # coefficients A matrix !!!!!
    def combine_constraints_coefficients(self):
        combine_constraints_coefficients = np.vstack(
            (inputTransform.embankment_constraints_coefficients_on_pavedArea(self), \
             inputTransform.embankment_constraints_coefficients_on_unPavedArea(self), \
             inputTransform.excavation_constraints_coefficients(self)
             )
        )
        return combine_constraints_coefficients

    # Refit
    def embankment_constraints_on_pavedArea(self):
        F_array = np.copy(self.F_matrix)
        return F_array

    # Refit
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
        N_array_extend[-1] = -self.exterior_inport
        return N_array_extend

    # coefficients b array !!!!!
    def combine_constraints(self):
        combine_constraints = np.hstack(
            (inputTransform.embankment_constraints_on_pavedArea(self), \
             inputTransform.embankment_constraints_on_unPavedArea(self), \
             inputTransform.excavation_constraints(self)
             ))
        return combine_constraints

    def objective_embankment_on_pavedArea(self):
        L_matrix = np.copy(self.L_matrix_refit())
        H_matrix = np.copy(self.H_matrix_refit())
        objective_embankment_on_pavedArea = L_matrix * H_matrix
        constraints_flat = [elem for row in objective_embankment_on_pavedArea for elem in row]
        return np.array(constraints_flat, dtype=float)

    def objective_embankment_on_unPavedArea(self):
        L_matrix = np.copy(self.L_matrix_refit())
        H_matrix = np.copy(self.H_matrix_refit())
        objective_embankment_on_pavedArea = L_matrix * H_matrix
        constraints_flat = [elem for row in objective_embankment_on_pavedArea for elem in row]
        return np.array(constraints_flat, dtype=float)

    def combine_objective(self):
        combine_objective = np.hstack((inputTransform.objective_embankment_on_pavedArea(self),
                                       inputTransform.objective_embankment_on_unPavedArea(self)))
        return combine_objective


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

        # #############for display only !!!!
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

        # #############for display only !!!!
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

        # #############for display only !!!!
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

        # #############for display only !!!!
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

                # #############for display only !!!!
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

                # #############for display only !!!!
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
    var_num = len(A_dual[0])
    solution = solution_row[var_num: -1]
    return optimal_value, solution

#创建实例
my_instance = inputTransform()

#Step1: 计算道面区填筑
c = my_instance.combine_objective()
A = my_instance.combine_constraints_coefficients()
b = my_instance.combine_constraints()

#输出道面区调配方式
num_of_instance = len(my_instance.P_coefficient_matrix())

paved_optimal_value, paved_solution = dual_simplex(c, A, b)
print("Optimal value:", paved_optimal_value)
print(f"Solution: \n {paved_solution.reshape(num_of_instance,(num_of_instance-1)*2 )}")