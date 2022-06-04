# HipID
Body pressure authentication by SSL

# Structure
```
HipID/
│
├── Baseline/
  ├── 8x8_Data/
    ├── create_dataset.py
    ├── denoising_diffusion_pytorch.py
    ├── generate_samples.py
    ├── ResNet18.py
    ├── split_csv_to_img.py
    └── train_model.py
├── raw_data/
├── moon_sojung/
├── park_sunghoon/
├── shin_dongwook/
├── ahn_hyungjun/
├── choi_cham/
└── choi_hyelim/
```


# 사용법

작업 디렉토리는 `8x8_Data`의 부모 디렉토리 기준입니다.

(이 경우 `팀원이름`) 

작업 디렉토리의 하위 디렉토리로 `8x8_Data`가 있고, 이름별로 데이터가 저장되어 있습니다.

1. `split_csv_to_img.py` 
    
    여러 data의 모임인 csv파일을 더 작은 2차원 csv파일로 분해합니다. 모듈을 실행시키면 현재 작업 디렉토리 아래의 모든 csv파일을 분해해 `some_name/images`와 `some_name/test_images`에 저장합니다.

    반복하여 실행되었을 때 이미 분해된 csv 파일을 분해하지 않도록 `images`, `samples`, `results`가 포함된 디렉토리에는 접근하지 않습니다.


2. `train_model.py`

    Diffusion 모델을 만들어 30000 step 동안 학습합니다. `run_exp`의 인자로 `8x8_Data` 아래의 `피험자ID`를 입력합니다.
    
    ```python
    if __name__ == "__main__":
        run_exp("0001")
    ```

    훈련 결과는 1000 step마다 `피험자ID/results`에 pt 파일로 저장됩니다.


3. `generate_samples.py`
   
    모델을 불러오고, 데이터를 생성합니다. `이름 폴더명`, `샘플 크기`, `csv로 저장 여부`, `normalize 여부`를 지정합니다.

    `csv로 저장 == True` 이면 생성된 데이터를(8,8)의 형태로  

    `작업 디렉토리/8x8_Data/피험자ID/gen_images/Gen_raw_1.csv`
     
     or

    `작업 디렉토리/8x8_Data/피험자ID/gen_images/Gen_unnormalized_1.csv`
     
    에 저장합니다.

4. `ResNet18.py`
    `Trainer`객체를 생성하고 데이터를 불러와 학습시킵니다. 사람을 분류하는 작업을 간편하게 학습할 수 있습니다.
    
    ```python
    from ResNet18 import Trainer


    trainer = Trainer(
        {"0001": 0, "0002": 1, "0003": 2, "0004": 3, "0005": 4},
        train_batch_size=32,
        num_classes=5,
    )
    # trainer.load(10, "name")  # 10번째 model을 불러와서 추가적으로 train_epoch만큼 훈련시킴

    trainer.train("cuda", pred_target="name")
    ```

    결과물은 `./backbone_results`에 저장되며, model만 불러올 경우 다음과 같이 사용합니다.

    ```python
    # some_path: path to data.pt 
    model = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            in_channels=1,
            num_classes=5,
        )
    
    model = nn.DataParallel(self.model.to(device))

    data = torch.load(
            some_path
            )

    model.load_state_dict(data['model'])
    
5. `barlow_twins.py`
   
   터미널에서 `python barlow_twins.py`를 실행합니다.
   사용하는 argument는 다음과 같습니다.
   
   * `--batch-size`: 배치 크기입니다

   * `--epochs`: 단계 수입니다

   * `--lambd`: loss에서 대각성분과 비대각 성분 사이의 가중치입니다.
 
   * `--projector`: representation을 embedding으로 만드는 projector의 차원 수입니다.
        
        (3개의 linear layer)

   * `--print-freq`: 몇 단계마다 loss 등 정보를 출력할지 설정합니다.

   * `--checkpoint-dir`: 결과물을 저장할 디렉토리를 설정합니다. 기본은 `./checkpoint/` 입니다.

   * `--backbone_weights`: 자세를 학습한 backbone이 저장된 path를 지정합니다. `./best_results/`아래의 경로를 지정하면 됩니다. 

        ex) `./best_results/backbone.pt` 가 존재하면
   `--backbone_weights=backbone.pt` 
   
# 작업 순서

## 1. csv 분해하기
`8x8_Data`의 상위 디렉토리로 이동합니다. 작업 디렉토리를 이 지점으로 고정합니다.

 `./8x8_Data/피험자_ID/` 별로 csv를 저장합니다. 작업 디렉토리에서 `split_csv_to_img.py`를 실행합니다. 

 `./8x8_Data/피험자_ID/images`와 `./8x8_Data/피험자_ID/test_images`에 8x8로 분해된 csv가 저장됩니다.

## 2. ResNet 학습시키기
작업 디렉토리에서 `ResNet18.py`를 실행시키면 다음 코드블럭이 실행됩니다.
```python
if __name__ == "__main__":
    trainer = Trainer(
        {"002": 0, "003": 1, "004": 2, "005": 3, "006": 4},
        train_batch_size=32,
        num_classes=5,  # 사람을 예측할 때는 사람의 수, pos 예측에는 8로 고정
        train_epochs=15,
        num_blocks=[2, 2, 2, 2],
    )
    # trainer.load(10, "name")  # 10번째 model을 불러와서 추가적으로 train_epoch만큼 훈련시킴
    trainer.train("cuda", pred_target="name")  # 자세를 예측할 때에는 pred_target='pos'
```
`Trainer`는 `ResNet`을 훈련시키기 위한 클래스로, 자세를 학습하도록 할 수도 있고, 이름을 학습하도록 할 수도 있습니다. 

공통적으로 `lookup_table={피험자_ID: label}`을 입력받습니다. `label`은 0부터 차례대로 입력해야 하며, 이름을 예측할 때만 사용됩니다. 자세 예측의 label은 파일명에서 자동으로 가져오기 때문에 따로 입력할 필요가 없습니다.

optimizer의 경우 `Trainer.train` 내에서 직접 변경하시면 됩니다.

`train_epoch`만큼의 학습을 진행하고, 매 epoch마다 `./backbone_results`에 pt파일을 저장합니다. 이미 학습시킨 모델을 불러올 경우, `Trainer.load(milestone, pred_target)`을 사용합니다. `milestone`에는 불러오려는 epoch을 입력하고, `pred_target`에는 학습 목표였던 `"name"` 혹은 `"pos"`를 입력합니다.

`Trainer.train(device, pred_target)`을 실행하면 설정한대로 훈련이 시작됩니다. 
모델을 불러온 경우, `train_epochs`는 추가적으로 수행할 epoch가 됩니다.

## 3. Barlow-Twins 학습

이 작업을 수행하기 전, 자세를 예측하도록 backbone을 훈련시킵니다. 그리고 가장 잘 나온 결과를 복사하여 `./best_results` 아래에 저장합니다.
`barlow_twins.py`를 실행하면 다음 코드블럭이 실행됩니다. lookup_table에 학습에 사용할 `피험자_ID`를 넣어줍니다.
```python
if __name__ == "__main__":
    main("cuda", {"001": 0})
```
이때, `barlow_twins.py`는 argparse를 통해 터미널로 인자를 받습니다. 세부 내용은 위의 사용법을 참고하세요.

예시
```
python.exe ./8x8_Data/barlow_twins.py --batch-size=512 --projector=1024-1024-1024 --epochs=1000 --print-freq=100 --backbone_weights=resnet18-pred-pos-8classes-10.pt
```

결과물은 `./checkpoint`에 저장됩니다.
