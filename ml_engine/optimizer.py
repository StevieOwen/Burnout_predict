import pulp

def solve_workload_redistribution(faculty_profiles: list, tasks_to_reallocate: list) -> list:
    """
    Uses Mixed-Integer Linear Programming (MILP via PuLP) to calculate optimal task 
    reassignments across department staff while adhering to safety capacity ceilings.
    """
    if not tasks_to_reallocate or not faculty_profiles:
        return []

    prob = pulp.LpProblem("UTB_Workload_Reallocation", pulp.LpMinimize)

    # Filter out faculty currently in HIGH risk tier
    eligible_staff = [f for f in faculty_profiles if f.get('risk_tier') != 'HIGH']
    if not eligible_staff:
        return []

    # Decision variables x[i][j]: task j assigned to staff i
    x = {}
    for task in tasks_to_reallocate:
        t_id = task['task_id']
        req_domain = task['required_domain']
        for staff in eligible_staff:
            s_id = staff['staff_id']
            # Check domain compatibility
            domains = [staff.get('primary_domain')] + (staff.get('secondary_domains') or [])
            is_qualified = 1 if req_domain in domains else 0
            x[(s_id, t_id)] = pulp.LpVariable(f"assign_{s_id}_{t_id}", cat=pulp.LpBinary)
            
            # Qualified domain constraint
            if not is_qualified:
                prob += x[(s_id, t_id)] == 0

    # Objective: Minimize total post-reallocation risk across eligible staff
    objective_terms = []
    for staff in eligible_staff:
        s_id = staff['staff_id']
        current_brs = staff['burnout_risk_score']
        
        # Estimate BRS impact of incoming tasks
        incoming_swu = pulp.lpSum([
            x[(s_id, t['task_id'])] * t['swu_value'] 
            for t in tasks_to_reallocate
        ])
        
        # Approximate BRS delta (1 SWU ~ 3.5 BRS points)
        projected_brs = current_brs + (incoming_swu * 3.5)
        
        # Safety Capacity Ceiling Constraint: Projected BRS <= 70.0
        prob += projected_brs <= 70.0, f"Max_Cap_{s_id}"
        objective_terms.append(projected_brs)

    prob += pulp.lpSum(objective_terms)

    # Task Coverage Constraint: Each task assigned to exactly one candidate
    for task in tasks_to_reallocate:
        t_id = task['task_id']
        prob += pulp.lpSum([x[(s['staff_id'], t_id)] for s in eligible_staff]) == 1, f"Cover_{t_id}"

    # Solve model using PuLP's CBC solver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    recommendations = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for task in tasks_to_reallocate:
            t_id = task['task_id']
            for staff in eligible_staff:
                s_id = staff['staff_id']
                if pulp.value(x[(s_id, t_id)]) == 1:
                    recommendations.append({
                        'task_id': t_id,
                        'task_type': task['task_type'],
                        'source_staff_id': task['source_staff_id'],
                        'recommended_target_staff_id': s_id,
                        'swu_transferred': task['swu_value']
                    })

    return recommendations