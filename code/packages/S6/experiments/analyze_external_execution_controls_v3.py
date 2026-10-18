"""Documented analysis-only amendment; frozen v2 runner and raw data untouched.

The native response shape has no phase. Recover it from its bound request intent
before the old analyzer's phase aggregation, without changing provider receipts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import analyze_external_execution_controls as original
from same_input_native_api import read as raw_read

def read_for_analysis(path):
    path=Path(path)
    value=raw_read(path)
    if path.parent.name=='responses':
        intent=raw_read(path.parent.parent/'intents'/path.name)
        assert intent['logical_id']==value['logical_id']
        assert intent['request_sha256']==value['request_sha256']
        assert intent['model_key']==value['model_key']
        assert intent['phase'] in ('selector','executor')
        value=value|{'phase':intent['phase']}
    return value

def build():
    old_read=original.read
    try:
        original.read=read_for_analysis
        value=original.build()
    finally:original.read=old_read
    value['version']='external-execution-controls-analysis-3'
    value['analysis_amendment']={'reason':'Native response has no phase; classify from bound immutable request intent',
      'original_runner_and_freeze_unchanged':True,'effect_based_protocol_change':False,
      'source_sha256':original.study.sha(Path(__file__))}
    return value

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    result=build()
    if args.write:
        (original.study.OUT/'analysis_v3.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        text=original.markdown(result).replace('external_execution_controls_20261005_v2/analysis.json','external_execution_controls_20261005_v2/analysis_v3.json').replace('| 状态一致 |','| 完整状态检查通过 |')
        text += '\n## 实际执行用量与配置边界\n\n'
        text += '| 模型 | 条件 | 实际executor请求 | prompt tokens | completion tokens | 回执reasoning tokens |\n|---|---|---:|---:|---:|---:|\n'
        for model, value in result['models'].items():
            for arm in ('D','U','P','A','E'):
                usage=value['usage_by_phase_arm'][arm]
                tokens=usage['recorded_tokens']
                text += f"| {model} | {arm} | {usage['recorded_receipts']} | {tokens['prompt_tokens']} | {tokens['completion_tokens']} | {tokens['reasoning_tokens']} |\n"
        text += '\n请求关闭reasoning不保证提供商内部没有推理；GLM仍报告非零reasoning tokens，保留实际回执而不按结果删臂。P按静态字节／槽位匹配，不按token数或自然干扰匹配；总用量还受到实际执行回合数影响。所有计划条件留存结果，不完整回复及过程失败保持原评分。\n'
        (original.study.ROOT/'docs/paper/external-execution-controls-results-20261005.md').write_text(text,encoding='utf8')
    print(json.dumps({'planned':360,'observed':result['observed_workflow_arms'],
      'attempts':{k:v['usage']['confirmed_attempts_with_receipt'] for k,v in result['models'].items()},
      'analysis_amendment':result['analysis_amendment']}))
