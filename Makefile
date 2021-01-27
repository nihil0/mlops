default: image test

push:
	@docker login
	@docker tag component-cond-test nihil0/mlops-demo
	@docker push nihil0/mlops-demo 

image:
	@bash scripts/build-test-image.sh

test:
	@docker run -v $(PWD):/build -v /tmp/azureml-models:/var/azureml-app/azureml-models component-cond-test pytest --disable-warnings
