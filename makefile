fmt:
	fd -e py --exclude object-detection/ | xargs isort --profile black
	fd -e py --exclude object-detection/ | xargs black

lint:
	fd -e py . src/ --exclude '*simmim*' --exclude '*moe*' | xargs flake8
