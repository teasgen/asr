from torchaudio.models.decoder import download_pretrained_files

def install_lm():
    files = download_pretrained_files("librispeech-4-gram")

    updated_lexicon = files.lexicon.replace("lexicon.txt", "lexicon_updated.txt")
    out = open(updated_lexicon, "w")
    with open(files.lexicon, "r") as f:
        lines = f.readlines()
        for x in lines:
            chars = x.split("\t")
            out.write(chars[0].replace("'", "") + "\t" + chars[1].replace(" '", ""))

    print(
        f"""
        Lexicon path: {updated_lexicon},
        Model path: {files.lm},
        """
    )

    return {
        "lexicon": updated_lexicon,
        "lm_path": files.lm,
    }