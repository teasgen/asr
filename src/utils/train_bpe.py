import argparse
import json
import os
import youtokentome as yttm

def merge_data_to_text_file(LS_index: str):
    result_filename = os.path.join(LS_index, "all_texts.txt")
    if os.path.exists(result_filename):
        return result_filename

    print("Processing index...")
    clean_100_index = os.path.join(LS_index, "train-clean-100_index.json")
    clean_360_index = os.path.join(LS_index, "train-clean-360_index.json")
    other_500_index = os.path.join(LS_index, "train-other-500_index.json")
    index = []
    index.extend(json.load(open(clean_100_index, "r")))
    index.extend(json.load(open(clean_360_index, "r")))
    index.extend(json.load(open(other_500_index, "r")))

    index = [x["text"] for x in index]

    with open(result_filename, "w") as f:
        for text in index:
            print(text, file=f)
    print(f"Finish index processing... Saved to {result_filename}")
    return result_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ls-indices-dir', type=str, help='path to directory with LS indices')
    parser.add_argument('--dir-to-save-model', type=str, help='path where to save bpe trained model')
    args = parser.parse_args()

    result_filename = merge_data_to_text_file(args.ls_indices_dir)

    train_data_path = result_filename 
    try:
        os.mkdir(args.dir_to_save_model)
        print(f"Directory '{args.dir_to_save_model}' created successfully.")
    except FileExistsError:
        print(f"Directory '{args.dir_to_save_model}' already exists.")

    model_path = os.path.join(args.dir_to_save_model, "ls_bpe.model")

    test_text = "I love you"

    # Training model
    yttm.BPE.train(data=train_data_path, vocab_size=100, model=model_path)

    # Loading model
    bpe = yttm.BPE(model=model_path)

    # Two types of tokenization
    print(bpe.encode([test_text], output_type=yttm.OutputType.ID))
    print(bpe.encode([test_text], output_type=yttm.OutputType.SUBWORD))
    