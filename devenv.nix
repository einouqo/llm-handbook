{pkgs, ...}: {
  packages = with pkgs; [
    mermaid-cli
  ];

  languages.typst.enable = true;
}
