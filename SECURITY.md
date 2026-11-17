# Security

TorchNEAT saves and loads checkpoints with `pickle`. Only load checkpoints (`.neat.pkl`) that you created or trust: loading a
malicious pickle can execute arbitrary code.

To report a vulnerability, use [GitHub's private vulnerability reporting](https://github.com/KshRaven/torchneat/security/advisories/new)
rather than a public issue. You will get a reply within a few days.
