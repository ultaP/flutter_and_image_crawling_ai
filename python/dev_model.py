from google.cloud import aiplatform

def create_training_pipeline_image_classification_sample(
    project: str,
    location: str,
    display_name: str,
    dataset_id: str,
    model_display_name: Optional[str] = None,
    model_type: str = "CLOUD",
    multi_label: bool = False,
    training_fraction_split: float = 0.8,
    validation_fraction_split: float = 0.1,
    test_fraction_split: float = 0.1,
    budget_milli_node_hours: int = 8000,
    disable_early_stopping: bool = False,
    sync: bool = True,
):
    # Google Cloud의 Vertex AI 초기화
    aiplatform.init(project=project, location=location)

    # AutoML 이미지 학습 작업 설정
    job = aiplatform.AutoMLImageTrainingJob(
        display_name=display_name,
        model_type=model_type,
        prediction_type="classification",
        multi_label=multi_label,
    )

    # 이미지 데이터셋 설정
    my_image_ds = aiplatform.ImageDataset(dataset_id)

    # 모델 학습 실행
    model = job.run(
        dataset=my_image_ds,
        model_display_name=model_display_name,
        training_fraction_split=training_fraction_split,
        validation_fraction_split=validation_fraction_split,
        test_fraction_split=test_fraction_split,
        budget_milli_node_hours=budget_milli_node_hours,
        disable_early_stopping=disable_early_stopping,
        sync=sync,
    )

    # 학습이 완료될 때까지 대기
    model.wait()

    # 학습된 모델의 정보 출력
    print(model.display_name)
    print(model.resource_name)
    print(model.uri)
    
    return model

# 사용 예시
create_training_pipeline_image_classification_sample(
    project="your-project-id",
    location="us-central1",
    display_name="my-image-classification-model",
    dataset_id="your-dataset-id",
    model_display_name="my-trained-model",
)
