#!/usr/bin/env python3
"""
フル実験の進捗監視スクリプト

使用方法:
    python monitor_experiment.py
"""

import time
import subprocess
import os
from datetime import datetime

def check_experiment_status():
    """実験の状況をチェック"""
    print(f"\n{'='*60}")
    print(f"🔍 実験状況チェック: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # プロセス確認
    try:
        result = subprocess.run(['pgrep', '-f', 'table_a2_parallel'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"✅ 実験プロセス実行中: PID {', '.join(pids)}")
            
            # CPU使用率確認
            for pid in pids:
                cpu_result = subprocess.run(['ps', '-p', pid, '-o', 'pcpu='], 
                                          capture_output=True, text=True)
                if cpu_result.stdout.strip():
                    cpu = cpu_result.stdout.strip()
                    print(f"   CPU使用率: {cpu}%")
        else:
            print("❌ 実験プロセスが見つかりません")
            return False
    except Exception as e:
        print(f"❌ プロセス確認エラー: {e}")
        return False
    
    # ログファイル確認
    if os.path.exists('full_experiment.log'):
        print(f"\n📄 最新ログ:")
        try:
            with open('full_experiment.log', 'r') as f:
                lines = f.readlines()
                # 最後の10行を表示
                for line in lines[-10:]:
                    if line.strip():
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"❌ ログ読み込みエラー: {e}")
    
    # 結果ファイル確認
    if os.path.exists('results'):
        print(f"\n📁 結果ファイル:")
        try:
            files = sorted([f for f in os.listdir('results') if f.endswith('.json')])
            for f in files[-3:]:  # 最新3ファイル
                path = f"results/{f}"
                size = os.path.getsize(path)
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                print(f"   {f}: {size:,}B ({mtime.strftime('%H:%M:%S')})")
        except Exception as e:
            print(f"❌ ファイル確認エラー: {e}")
    
    # 完了通知ファイル確認
    if os.path.exists('results/completion_notification.txt'):
        print(f"\n🎉 実験完了！")
        with open('results/completion_notification.txt', 'r') as f:
            print(f.read())
        return False
    
    return True

def main():
    """メイン監視ループ"""
    print("🚀 実験監視開始")
    print("停止するには Ctrl+C を押してください")
    
    try:
        while True:
            if not check_experiment_status():
                print("\n🏁 実験が完了または停止しました")
                break
            
            print(f"\n⏰ 次のチェック: 5分後")
            time.sleep(300)  # 5分待機
            
    except KeyboardInterrupt:
        print("\n\n👋 監視を停止しました")

if __name__ == "__main__":
    main() 