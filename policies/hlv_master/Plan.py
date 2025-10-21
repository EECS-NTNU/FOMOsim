from .Simple_calculations import copy_arr_iter

class Plan():
    def __init__(self, copied_plan, tabu_list, weight_set = None, branch_number = None):
        self.plan = copied_plan
        self.next_visit = None
        self.find_next_visit()
        self.tabu_list = tabu_list
        self.weight_set = weight_set
        self.branch_number = branch_number


    def find_next_visit(self):
        arrival_time = float('inf')
        for vehicle_id in self.plan:
            if self.plan[vehicle_id][-1].arrival_time < arrival_time:
                arrival_time = self.plan[vehicle_id][-1].arrival_time
                self.next_visit = self.plan[vehicle_id][-1]

    def copy_plan(self):
        plan_copy = dict()
        for vehicle in self.plan:
            route_copy = copy_arr_iter(self.plan[vehicle])
            plan_copy[vehicle] = route_copy
        return plan_copy
    
    def __repr__(self) -> str:
        return f'Plan: {self.plan}, next_visit = {self.next_visit}'