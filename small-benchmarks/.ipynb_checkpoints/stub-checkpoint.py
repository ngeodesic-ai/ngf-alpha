# Generate 100 synthetic ARC tasks with varied transformations
def generate_arc_task():
    grid = [[random.randint(1, 9) for _ in range(random.choice([2, 3]))] for _ in range(random.choice([2, 3]))]
    transform_type = random.choice(['rotate', 'flip_h', 'flip_v', 'scale', 'multi_step', 'swap_colors', 'shift'])
    if transform_type == 'rotate':
        if len(grid) == 2:
            output = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
        else:
            output = [grid[2], grid[1], grid[0]]
        desc = "(90 deg rotate)"
    elif transform_type == 'flip_h':
        output = [row[::-1] for row in grid]
        desc = "(horizontal flip)"
    elif transform_type == 'flip_v':
        output = grid[::-1]
        desc = "(vertical flip)"
    elif transform_type == 'scale':
        output = [[x * 2 for x in row] for row in grid]
        desc = "(scale by 2)"
    elif transform_type == 'multi_step':
        rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]] if len(grid) == 2 else [grid[2], grid[1], grid[0]]
        output = [row[::-1] for row in rotated]
        desc = "(rotate then flip)"
    elif transform_type == 'swap_colors':
        flat = [item for sublist in grid for item in sublist]
        if flat:
            max_val = max(flat)
            min_val = min(flat)
            output = [[max_val if x == min_val else min_val if x == max_val else x for x in row] for row in grid]
        desc = "(swap max/min values)"
    else:
        output = grid[1:] + [grid[0]]
        desc = "(circular shift)"
    prompt = f"Identify the pattern: Input grid {grid} -> Output {output} {desc}. Apply to {grid}."
    correct_example = f"Apply to {grid} results in {output} {desc}."
    return prompt, output, correct_example

arc_tasks = [generate_arc_task() for _ in range(100)]