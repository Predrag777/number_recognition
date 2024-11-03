<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $data = $_POST['imageData'];
    $decodedData = base64_decode(str_replace('data:image/png;base64,', '', $data));
    
    $folderPath = 'izlazne_slike/';
    $filename = $folderPath . 'slika.png';
    
    if (!file_exists($folderPath)) {
        mkdir($folderPath, 0777, true); // Pravimo folder ako ne postoji
    }

    if (file_put_contents($filename, $decodedData)) {
        echo 'Slika je uspešno sačuvana.';
    } else {
        echo 'Došlo je do greške prilikom čuvanja slike.';
    }


    
}
$pythonScript = "pajton/read_image.py"; // Putanja do vaše Python skripte
$output = shell_exec("pajton/venv/bin/python3 $pythonScript 2>&1");
echo $output;

?>

