import json
correct = 0
total = 0
with open('PACS_data/inference_multi_video_json/test_data_mat_result_3B_only_PACS_m.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'][0] == data['output'][0]:
            correct += 1
        total += 1
    print(f'准确率: {correct/total}')