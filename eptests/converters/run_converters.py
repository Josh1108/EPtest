from eptests.converters.iface import (
    Spr1,
    Spr2,
    Rel,
    Dpr,
    DepEwt,
    OntonotesConstituent,
    OntonotesSrl,
    OntonotesNer,
    OntonotesCoref,
    OntonotesPos,
    ConlluPOS,
    Conll2003NER,
    Conll2000Chunking,
    SyntacticDependencyClasssificationEwt,
    SyntacticDepenedencyPredictionEwt,
    SyntacticDependencyClassificationPenn,
    SyntacticDependencyPredictionPenn,
    PosPenn,
)


import argparse
import pickle
import os


def run_tasks(args):
    converter = None
    if args.task == "spr2":
        converter = Spr2(
            args.path, args.task, "edges.train.json", "edges.test.json", "edges.dev.json"
        ).run()

    elif args.task == "rel-semeval-2010-task8":
        converter = Rel(args.path, args.task, "train.json", "test.json", dev_set=False).run()

    elif args.task == "dpr":
        converter = Dpr(args.path, args.task, "train.json", "test.json", "dev.json").run()

    elif args.task == "dep-ewt":
        converter = DepEwt(
            args.path,
            args.task,
            "en_ewt-ud-train.json",
            "en_ewt-ud-test.json",
            "en_ewt-ud-dev.json",
        ).run()

    elif args.task == "ontonotes-const":
        converter = OntonotesConstituent(
            args.path, args.task, "train.json", "test.json", "development.json"
        ).run()

    elif args.task == "ontonotes-pos":
        converter = OntonotesPos(
            args.path, args.task, "train.json", "test.json", "development.json"
        ).run()

    elif args.task == "ontonotes-coref":
        converter = OntonotesCoref(
            args.path, args.task, "train.json", "test.json", "development.json"
        ).run()

    elif args.task == "ontonotes-ner":
        converter = OntonotesNer(
            args.path, args.task, "train.json", "test.json", "development.json"
        ).run()

    elif args.task == "ontonotes-srl":
        converter = OntonotesSrl(
            args.path, args.task, "train.json", "test.json", "development.json"
        ).run()

    elif args.task == "conll-2000-chunking":
        converter = Conll2000Chunking(
            args.path, args.task, "train.txt", "test.txt", dev_set=False
        ).run()

    elif args.task == "ewt-pos":
        converter = ConlluPOS(
            args.path,
            args.task,
            "en_ewt-ud-train.conllu",
            "en_ewt-ud-test.conllu",
            "en_ewt-ud-dev.conllu",
        ).run()
    elif args.task == "ptb-pos":
        converter = PosPenn(args.path, args.task, "train.conllx", "test.conllx", "dev.conllx").run()
    elif args.task == "ewt-syn-dep-pred":
        converter = SyntacticDepenedencyPredictionEwt(
            args.path,
            args.task,
            "en_ewt-ud-train.conllu",
            "en_ewt-ud-test.conllu",
            "en_ewt-ud-dev.conllu",
        ).run()
    elif args.task == "ewt-syn-dep-cls":
        converter = SyntacticDependencyClasssificationEwt(
            args.path,
            args.task,
            "en_ewt-ud-train.conllu",
            "en_ewt-ud-test.conllu",
            "en_ewt-ud-dev.conllu",
        ).run()
    elif args.task == "ptb-syn-dep-pred":
        converter = SyntacticDependencyPredictionPenn(
            args.path, args.task, "train.conllx", "test.conllx", "dev.conllx"
        ).run()
    elif args.task == "ptb-syn-dep-cls":
        converter = SyntacticDependencyClassificationPenn(
            args.path, args.task, "train.conllx", "test.conllx", "dev.conllx"
        ).run()
    elif args.task == "spr1":
        converter = Spr1(
            args.path, args.task, "spr1.train.json", "spr1.test.json", "spr1.dev.json"
        ).run()
    elif args.task == "conll-2003-ner":
        converter = Conll2003NER(
            args.path,
            args.task,
            "conll-2003.train.txt",
            "conll-2003.test.txt",
            "conll-2003.dev.txt",
        ).run()
    # elif args.task =='opensdp-sem-dep-cls':
    # converter = SemanticDependencyClassificationOpenSDP(args.path,args.task).run()
    if converter is not None:
        with open(os.path.join(args.out_dir, f"{args.task}.pickle"), "wb") as f:
            pickle.dump(converter, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/jushaan/Projects/probe/data/edges/spr2")
    parser.add_argument("--task", default="spr2")
    parser.add_argument("--out_dir", default="/home/jushaan/Projects/probe/data/EPdata/")
    args = parser.parse_args()
    run_tasks(args)
