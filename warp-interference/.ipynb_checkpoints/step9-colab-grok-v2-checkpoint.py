# Step 9: Benchmark on 10 Synthetic ARC Tasks
import random

def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    # Simulate 90-degree rotation
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
    return grid, rotated

tasks = [generate_arc_task() for _ in range(10)]
stock_correct = 0
warped_correct = 0
stock_hallucinations = 0
warped_hallucinations = 0

for i, (input_grid, expected_output) in enumerate(tasks, 1):
    prompt = f"Identify the pattern: Input grid {input_grid} -> Output {expected_output} (90 deg rotate). Apply to {input_grid}."
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Stock GPT-2
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)
    stock_output = tokenizer.decode(torch.argmax(outputs.logits, dim=-1)[0])
    stock_correct += 1 if str(expected_output) in stock_output else 0
    stock_hallucinations += 1 if any(c not in "0123456789[], " for c in stock_output) else 0
    
    # Warped Geodesic
    generated = inputs['input_ids']
    past_key_values = None
    for _ in range(20):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past_key_values)
                outputs = model(generated, past_key_values=cache, output_hidden_states=True, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        next_token = torch.clamp(next_token, 0, vocab_size - 1)
        generated = torch.cat([generated, next_token], dim=1).to(device)
        
        if generated.shape[1] % 5 == 0:
            current_hidden = outputs.hidden_states[-1][:, -1, :].to(device)
            current_latent = current_hidden.cpu().numpy()
            reduced_current = pca.transform(current_latent)
            nudged_reduced = symbolic_nudge(reduced_current[0], nudge_target)
            nudged_latent = pca.inverse_transform(nudged_reduced.reshape(1, -1))[0]
            norm = np.linalg.norm(nudged_latent)
            if norm > 0:
                nudged_latent = (nudged_latent / norm) * 5.0
            nudged_hidden = torch.from_numpy(nudged_latent).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
            if nudged_logits.size(0) > vocab_size:
                nudged_logits[vocab_size:] = -float('inf')
            next_token = torch.argmax(nudged_logits, dim=-1).unsqueeze(0)
            next_token = torch.clamp(next_token, 0, vocab_size - 1)
            if next_token.item() == 0 or next_token.item() >= vocab_size:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            past_key_values = None
            generated = torch.cat([generated[:, :-1], next_token], dim=1).to(device)
    
    warped_output = tokenizer.decode(generated[0]).lower()
    warped_correct += 1 if str(expected_output).lower() in warped_output else 0
    warped_hallucinations += 1 if any(c not in "0123456789[], " for c in warped_output) else 0
    print(f"Task {i}: Stock '{stock_output[:50]}...', Warped '{warped_output[:50]}...', Expected {expected_output}")

# Summary
stock_acc = stock_correct / 10 * 100
warped_acc = warped_correct / 10 * 100
hall_red = ((stock_hallucinations - warped_hallucinations) / stock_hallucinations * 100) if stock_hallucinations > 0 else 0
print(f"\nBenchmark Results:")
print(f"Stock Accuracy: {stock_acc:.1f}%, Warped Accuracy: {warped_acc:.1f}%, Hallucination Reduction: {hall_red:.1f}%")