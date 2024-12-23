FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY . ${LAMBDA_TASK_ROOT}
RUN chmod -R o+rX .
# Install the specified packages
RUN pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]