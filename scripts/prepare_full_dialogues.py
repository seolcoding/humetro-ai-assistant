#!/usr/bin/env python3
"""
Prepare Full Dialogues Dataset
================================

Combine all training dialogues from Dasan call center into one file
for full knowledge extraction.
"""

import json
from pathlib import Path
from datetime import datetime


def main():
    source_dir = Path("data/AI_HUB_DASAN_QA/01_extracted/training/labeled")
    output_file = Path("data/processed/knowledge_extraction/full_dialogues.json")

    categories = [
        {
            "name": "대중교통_안내",
            "file": "민원(콜센터) 질의응답_다산콜센터_대중교통 안내_Training.json"
        },
        {
            "name": "생활하수도_관련_문의",
            "file": "민원(콜센터) 질의응답_다산콜센터_생활하수도 관련 문의_Training.json"
        },
        {
            "name": "일반행정_문의",
            "file": "민원(콜센터) 질의응답_다산콜센터_일반행정 문의_Training.json"
        },
        {
            "name": "코로나19_관련_상담",
            "file": "민원(콜센터) 질의응답_다산콜센터_코로나19 관련 상담_Training.json"
        }
    ]

    print("="*40)
    print("📚 PREPARING FULL DIALOGUES DATASET")
    print("="*40)

    all_dialogues = {}
    total_dialogues = 0
    total_turns = 0

    for category_info in categories:
        category_name = category_info["name"]
        file_path = source_dir / category_info["file"]

        print(f"\n📂 Loading {category_name}...")
        print(f"   File: {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            flat_data = json.load(f)

        # Group by dialogue ID
        dialogues_dict = {}
        for turn_data in flat_data:
            dialogue_id = turn_data["대화셋일련번호"]

            if dialogue_id not in dialogues_dict:
                dialogues_dict[dialogue_id] = []

            dialogues_dict[dialogue_id].append(turn_data)

        # Convert to same format as sampled_dialogues.json
        formatted_dialogues = []
        category_turns = 0

        for dialogue_id, turns in dialogues_dict.items():
            formatted_dialogue = {
                "dialogue_id": dialogue_id,
                "category": category_name,
                "total_turns": len(turns),
                "turns": []
            }

            for turn_data in turns:
                speaker = turn_data["화자"]
                turn_num = int(turn_data["문장번호"])
                qa_type = turn_data["QA"]

                # Get text based on speaker and QA type
                if speaker == "고객":
                    if qa_type == "Q":
                        text = turn_data["고객질문(요청)"]
                    else:
                        text = turn_data["고객답변"]
                else:  # 상담사
                    if qa_type == "Q":
                        text = turn_data["상담사질문(요청)"]
                    else:
                        text = turn_data["상담사답변"]

                # Get intent
                intent = turn_data["고객의도"] if speaker == "고객" else turn_data["상담사의도"]

                # Parse entities and terminology
                entities_str = turn_data.get("개체명 ", "").strip()
                entities = [e.strip() for e in entities_str.split(",") if e.strip()] if entities_str else []

                terminology_str = turn_data.get("용어사전", "").strip()
                terminology = {}
                if terminology_str:
                    for term_info in terminology_str.split("/"):
                        parts = term_info.strip().split("/")
                        if len(parts) >= 2:
                            terminology[parts[0]] = parts[1]

                kb_tags_str = turn_data.get("지식베이스", "").strip()
                kb_tags = [t.strip() for t in kb_tags_str.split(",") if t.strip()] if kb_tags_str else []

                formatted_turn = {
                    "turn_num": turn_num,
                    "speaker": speaker,
                    "qa_type": qa_type,
                    "text": text,
                    "intent": intent,
                    "entities": entities,
                    "terminology": terminology,
                    "kb_tags": kb_tags
                }
                formatted_dialogue["turns"].append(formatted_turn)

            formatted_dialogues.append(formatted_dialogue)
            category_turns += len(turns)

        all_dialogues[category_name] = formatted_dialogues
        total_dialogues += len(formatted_dialogues)
        total_turns += category_turns

        print(f"   ✅ Loaded: {len(formatted_dialogues)} dialogues, {category_turns} turns")

    # Create metadata
    metadata = {
        "description": "Full Dasan Call Center training dialogues for knowledge extraction",
        "source_dir": str(source_dir),
        "created_at": datetime.now().isoformat(),
        "stats": {
            "total_dialogues": total_dialogues,
            "total_turns": total_turns,
            "categories": {}
        }
    }

    for category_name, dialogues in all_dialogues.items():
        category_turns = sum(d["total_turns"] for d in dialogues)
        metadata["stats"]["categories"][category_name] = {
            "dialogues": len(dialogues),
            "turns": category_turns,
            "avg_turns": round(category_turns / len(dialogues), 2) if dialogues else 0
        }

    # Save combined file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": metadata,
        "dialogues": all_dialogues
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"✅ DATASET PREPARED")
    print(f"{'='*40}")
    print(f"📊 Total dialogues: {total_dialogues:,}")
    print(f"📊 Total turns: {total_turns:,}")
    print(f"📊 Average turns per dialogue: {total_turns / total_dialogues:.2f}")
    print(f"\n💾 Output: {output_file}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
