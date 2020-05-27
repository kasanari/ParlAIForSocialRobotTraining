import json
import csv
import argparse

parser = argparse.ArgumentParser(description='Convert ParlAI json output to csv.')

parser.add_argument('models', type=str, nargs='+', help='model names')
args = parser.parse_args()

files = args.models

for filename in files:

    filename = filename[len("eval/"):]

    with open(f"eval_results/{filename}_neil_replies.jsonl") as f:
        lines = f.readlines()

    with open(f"eval_results/{filename}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "situation", "affect", "utterance"], extrasaction="ignore")
        writer.writeheader()
        for line in lines:
            data = json.loads(line)['dialog'][0]
            to_write = {
                "scenario" : data[0]['scenario'],
                "situation" : data[0]['situation'],
                "affect" : data[0]['emotion'],
                "utterance" : data[1]['text'],
            }
            writer.writerow(to_write)