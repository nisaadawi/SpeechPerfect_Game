<?php
header('Content-Type: application/json');

$servername = "localhost";
$username = "humancmt_nd_admin";
$password = "Wereenz0909?";
$dbname = "humancmt_nd_speechperfect";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode(["error" => "Connection failed"]);
    exit();
}

$sql = "SELECT name FROM players ORDER BY id DESC LIMIT 10";
$result = $conn->query($sql);

$names = [];

if ($result->num_rows > 0) {
    while($row = $result->fetch_assoc()) {
        $names[] = $row["name"];
    }
}

$conn->close();

echo json_encode($names);
?>
