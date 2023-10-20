task_names=("conll-2000-chunking" "conll-2003-ner" "dpr" "ewt-pos" "ewt-syn-dep-cls" "ewt-syn-dep-pred" "ontonotes-const" "ontonotes-coref" "ontonotes-ner" "ontonotes-pos" "ontonotes-srl" "ptb-pos" "ptb-syn-dep-cls" "ptb-syn-dep-pred" "semeval-2010-task8-rel-cls" "spr1" "spr2")
for task_name in "${task_names[@]}"
do
    echo "$task_name"
    python create-config.py --new_task $task_name --model bertbc
    python create-config.py --new_task $task_name --model rb
done