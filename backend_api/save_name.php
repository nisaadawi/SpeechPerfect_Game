<?php
$servername = "localhost";
$username = "humancmt_nd_admin";
$password = "Wereenz0909?";
$dbname = "humancmt_nd_speechperfect";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    http_response_code(500);
    echo "Connection failed: " . $conn->connect_error;
    exit();
}

// Get POST data
$name = $_POST['name'] ?? '';

if ($name != '') {
    $stmt = $conn->prepare("INSERT INTO players (name) VALUES (?)");
    $stmt->bind_param("s", $name);
    $stmt->execute();
    $stmt->close();
    echo "Name saved!";
} else {
    http_response_code(400);
    echo "Invalid name.";
}

$conn->close();
?>
