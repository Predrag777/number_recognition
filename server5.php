<?php
$pythonScript = "pajton/train_network.py"; // Putanja do vaše Python skripte
$output = shell_exec("pajton/venv/bin/python3 $pythonScript 2>&1");
echo $output;
?>