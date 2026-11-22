[English](README.md) | 日本語

# llama.cpp-spin8

**Spin8 G128量子化を扱う実験用llama.cpp拡張。** 圧縮したGGUF重みを、専用のCPU/CUDA演算とQwen35グラフで実行します。量子化、直接重み編集、指定方向の射影、ColabでのHF取得・変換・評価のコードを含みます。

基盤は [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) のコミット [`7fce5439fe451cd48340b112ec37d2befe02c776`](https://github.com/ggml-org/llama.cpp/commit/7fce5439fe451cd48340b112ec37d2befe02c776) です。元の案内は [上流READMEの保存版](docs/spin8/UPSTREAM_README.md) を参照してください。

## 現在の到達点

**2026-11-23（日本時間）時点の記録です。**

| 項目 | 状況 |
| --- | --- |
| CPU/CUDAのSpin8演算、ローダー、Qwen35グラフ接続 | 実装あり |
| 合成小型モデルのCPU推論・射影付き推論 | 検証済み |
| HF取得、変換、再開、評価プログラム | 実装と小型検証あり |
| ネイティブCPU/CUDA codec、通常・射影つきgraph parity、生成API | Colab A100-SXM4-40GBで合格 |
| Qwen3.8-27Bの元重みの取得 | 全18シャードを取得・検証済み |
| 27B全体のSpin8量子化 | 実行中 |
| 完成27B GGUFの推論・数学ベンチマーク・HF公開 | 未完了 |

今回のA100実行はPyTorch `2.14.1+cu130`、CUDA `13.0`を使用しました。後述するFP32比較条件での最大logit絶対誤差は、通常モデルで `3.4570693969726562e-6`、射影つきモデルで `3.427267074584961e-6` です。従来の `atol=rtol=1e-4` を変更せずに両方が合格し、`MUL_MAT_SPIN8` のCUDA実行配置も確認しました。

検証条件と成果物のハッシュは [STATUS.md](docs/spin8/STATUS.md) に記録しています。この比較は、合成小型モデルと、同じSpin8重みを復号したCPU参照を使った演算検証です。元のBF16モデルに対する品質評価ではありません。学習済み27B重みや、数学能力・拒否率の改善、モデル全体の推論速度を示す完了済みの実測値は含みません。

## 形式と対応範囲

- 128重みごとに34バイトを使い、**圧縮payloadは2.125 bits/weight**です。行列ごとの回転bank、GGUFメタデータ、圧縮対象外のテンソルが別に必要です。
- Hadamard変換と8次元回転bankを使います。既定のtriality bankはseedから再現される固定bankです。回転の追加学習や、三つのLLMによる推論は行いません。
- GGML type **44**、file type **43**、codec version **1** は、この固定ランタイム専用の未登録拡張です。上流llama.cpp、Ollama、LM Studioの標準ローダーとの互換性はありません。
- 現行の変換対象は、正規変換済みの `qwen35` GGUFです。対応するCPU/CUDAの圧縮行列積・埋め込み参照を使用します。全モデル系列への一般対応を確認したものではありません。
- 数学ドナーとの差分と、入力された方向の射影を任意で適用できます。方向を自動発見する機能はなく、その意味や実モデルでの効果は別途評価が必要です。

詳しい仕様は [Spin8の説明](tools/spin8/README.md) と [検証手順](tools/spin8/TESTING.md) にあります。

## CPUでビルドして小型モデルを確認

Linux、CMake、C/C++コンパイラ、Pythonを用意し、リポジトリを取得します。

```sh
git clone https://github.com/zapabob/llama.cpp-spin8.git
cd llama.cpp-spin8
```

リポジトリのルートで実行します。以下は外部Web UIアセットを取得しないAPIサーバー構成です。

```sh
cmake -S . -B build-spin8-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_RPC=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-spin8-cpu --target llama-server test-spin8-native -j 4

python tools/spin8/verify_native.py \
  --binary build-spin8-cpu/bin/test-spin8-native \
  --model examples/spin8-colab/payload/fixtures/tiny-qwen35-spin8.gguf \
  --control examples/spin8-colab/payload/fixtures/tiny-qwen35-decoded.gguf \
  --out-dir build-spin8-cpu/spin8-parity \
  --ngl 0 --control-ngl 0 --cache-type f32 \
  --atol 1e-4 --rtol 1e-4 --require-spin8-device CPU
```

この比較は、同じSpin8重みを復号したF32モデルとの演算一致を確認します。元の学習済みモデルに対する量子化精度や文章品質の評価とは別です。

## A100用CUDAビルド

CUDA Toolkitと対応コンパイラを用意し、CPU版とは別のビルドディレクトリを使います。`80` はA100用の指定です。

```sh
cmake -S . -B build-spin8-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGGML_NATIVE=OFF -DGGML_RPC=OFF \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_FA=OFF -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-spin8-cuda --target llama-server test-spin8-native -j 4

GGML_CUDA_SPIN8_VALIDATE_F32=1 NVIDIA_TF32_OVERRIDE=0 \
python tools/spin8/verify_native.py \
  --binary build-spin8-cuda/bin/test-spin8-native \
  --model examples/spin8-colab/payload/fixtures/tiny-qwen35-spin8.gguf \
  --control examples/spin8-colab/payload/fixtures/tiny-qwen35-decoded.gguf \
  --out-dir build-spin8-cuda/spin8-parity \
  --ngl 99 --control-ngl 0 --cache-type f32 \
  --atol 1e-4 --rtol 1e-4 \
  --require-spin8-device CUDA --require-spin8-op MUL_MAT_SPIN8
```

これは今回のA100による合成モデル比較で使用した条件です。両モデルのK/VをF32で保持し、キャッシュの精度を揃えます。比較用の `GGML_CUDA_SPIN8_VALIDATE_F32` は通常のF32行列積をcuBLASのF32経路へ送り、独自MMF経路のTF32命令を避けます。`NVIDIA_TF32_OVERRIDE=0` はcuBLAS側のTF32を無効にします。Spin8カーネルは変更せず、設定はこの比較プロセスだけに適用します。Colabのbootstrapは条件を記録し、生成APIの確認には通常の環境を使います。

配置検査は通常のCPUフォールバックを検出しますが、速度ベンチマークやGPUカーネルプロファイラではありません。射影つきモデルの確認と記録項目は [検証手順](tools/spin8/TESTING.md) を参照してください。

## 量子化と直接重み編集

通常のfloat GGUFからの量子化は [convert.py](tools/spin8/convert.py)、数学差分・方向射影は [edit_weights.py](tools/spin8/edit_weights.py) を使います。Pythonの依存と入力条件は [Spin8の説明](tools/spin8/README.md) を参照してください。CUDA encoderにはCUDA版PyTorchが必要です。

```sh
python tools/spin8/convert.py quantize model-f16.gguf model-spin8.gguf \
  --include-embedding --encoder torch --device cuda \
  --compute-dtype float32 --chunk-groups 8192 \
  --resume-dir spin8-cache/run-01

build-spin8-cuda/bin/llama-server -m model-spin8.gguf \
  --host 127.0.0.1 --port 8080 -ngl -1
```

元モデル、変換条件、変更した重みの由来を保存してください。方向射影は指定ベクトルに対する線形操作であり、拒否の除去や数学能力向上を保証するものではありません。回答データを使うSFTやoptimizer更新は、この重み編集パイプラインには入っていません。

## Colabとベンチマーク

[Colab用ノートブック](examples/spin8-colab/Spin8_A100_Weight_Edit.ipynb) と [日本語の実行手順](examples/spin8-colab/README_JA.md) を同梱しています。

[Google Colabで開く](https://colab.research.google.com/github/zapabob/llama.cpp-spin8/blob/main/examples/spin8-colab/Spin8_A100_Weight_Edit.ipynb)

この公開リポジトリのノートブックをColabで開けます。ノートブックをダウンロードしてColabへアップロードする方法も使えます。

ノートブックは検証済みのソースpayloadを内蔵し、上流の固定コミットへ同梱パッチを適用します。リポジトリの将来の `main` を自動的に追従する仕組みではありません。

大きな処理の前に実際のGPUを確認し、ColabがA100以外を割り当てた場合は停止します。既存のPyTorch/CUDAを維持し、処理中の出力を逐次表示します。符号化の既定値は `chunk_groups=8192`、`tile_elements=1048576`、`scale_steps=17`、seed `81237` です。

初期設定は通常のSpin8変換とGSM8Kの64問評価です。数学差分は `MATH_DELTA=True`、方向射影はモデルに対応した実ファイルを `DIRECTION_PATH` に指定して有効にします。全1,319問は `BENCHMARK_FULL=True` で選びます。元モデルF16との比較には追加のディスク容量が必要で、比較対象を用意できない場合は改善を未測定として記録します。拒否率ベンチマークは含みません。

## ライセンスと由来

[MIT License](LICENSE)。上流の著作権表示とライセンスを保持しています。基盤となるllama.cpp/ggmlはThe ggml authorsによるものです。Spin8拡張の設計・実装・検証にはAI支援を使用しています。モデル重みの取得・配布には、各モデルのライセンスが別途適用されます。
