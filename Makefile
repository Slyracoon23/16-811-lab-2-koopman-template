.PHONY: check reproduce extend clean

check:            ## Run the tests. The failures are your to-do list.
	pytest -q

reproduce:        ## Run across seeds, write results.json
	python3 reproduce.py --seeds 5

extend:           ## Your ideas, one row each, into extensions.json
	python3 extend.py --all --seeds 5

clean:
	rm -f results.json extensions.json && find . -name __pycache__ -type d -exec rm -rf {} +
