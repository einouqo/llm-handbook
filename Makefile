.PHONY: html pdf clean

SITE_DIR ?= site
DIST_DIR ?= dist

html:
	@rm -rf "$(SITE_DIR)"
	@mkdir -p "$(SITE_DIR)"
	@typst compile --features html main.typ "$(SITE_DIR)/index.html" -f html
	@cp -R assets "$(SITE_DIR)/assets" || true

pdf:
	@rm -rf "$(DIST_DIR)"
	@mkdir -p "$(DIST_DIR)"
	@typst compile main.typ "$(DIST_DIR)/llm-handbook.pdf" -f pdf

clean:
	@rm -rf "$(SITE_DIR)" "$(DIST_DIR)"

