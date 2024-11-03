<?php
    $pythonScript = "pajton/main.py"; // Putanja do vaše Python skripte
    $output = shell_exec("pajton/venv/bin/python3 $pythonScript 2>&1");
    $string = $output;
    $delimiter = "Nacrtao si:";

    // Deljenje stringa na osnovu delimitera "BROJ"
    $parts = explode($delimiter, $string);

    // Uklanjanje prvog praznog elementa u nizu
    array_shift($parts);

    // Prikazivanje delova stringa koji počinju sa "BROJ"
    foreach ($parts as $part) {
        $broj = preg_replace('/\D/', '', $part); // Uzimanje samo brojeva
        echo $broj;
    }

?>