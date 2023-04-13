DOCKER := docker
build:
	${DOCKER} build -t myapp .
build-nc:
	${DOCKER} build --no-cache -t myapp .
run:
	${DOCKER} run -p 80:80 myapp
