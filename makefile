fmt:
	fd . src -e py | xargs isort --profile black
	fd . src -e py | xargs black

lint:
	fd -e py . src/ --exclude '*simmim*' --exclude '*moe*' | xargs flake8
