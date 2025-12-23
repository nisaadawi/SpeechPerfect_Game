<?php
/**
 * Create a new session and return session_id
 * This should be called at the start of a game session
 */
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

require_once 'db_config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["success" => false, "message" => "Only POST method allowed"]);
    exit();
}

$input = json_decode(file_get_contents('php://input'), true);

// Debug logging
error_log("create_session.php - Received input: " . json_encode($input));

$user_id = intval($input['user_id'] ?? 0);

if (!$user_id) {
    $error_msg = "Missing or invalid user_id. Received: " . json_encode($input);
    error_log("create_session.php - " . $error_msg);
    echo json_encode(["success" => false, "message" => $error_msg]);
    exit();
}

try {
    // Verify table exists and check structure
    $check_table = $pdo->query("SHOW TABLES LIKE 'tbl_session'");
    if ($check_table->rowCount() == 0) {
        echo json_encode([
            "success" => false,
            "message" => "Table tbl_session does not exist. Please run create_tbl_session.sql first."
        ]);
        exit();
    }
    
    // Verify columns exist
    $columns = $pdo->query("SHOW COLUMNS FROM tbl_session")->fetchAll(PDO::FETCH_COLUMN);
    error_log("create_session.php - tbl_session columns: " . json_encode($columns));
    
    if (!in_array('id', $columns)) {
        echo json_encode([
            "success" => false,
            "message" => "Column 'id' does not exist in tbl_session. Available columns: " . implode(', ', $columns)
        ]);
        exit();
    }
    
    // Verify session_id is AUTO_INCREMENT
    $auto_increment_check = $pdo->query("SHOW COLUMNS FROM tbl_session WHERE Field = 'session_id' AND Extra LIKE '%auto_increment%'");
    if ($auto_increment_check->rowCount() == 0) {
        echo json_encode([
            "success" => false,
            "message" => "session_id column is not set to AUTO_INCREMENT. Please run: ALTER TABLE tbl_session MODIFY session_id INT(11) NOT NULL AUTO_INCREMENT;"
        ]);
        exit();
    }
    
    // Create new session record
    // Note: session_id is AUTO_INCREMENT, so we don't insert it
    // timestamp has DEFAULT CURRENT_TIMESTAMP, but we'll set it explicitly
    $stmt = $pdo->prepare("
        INSERT INTO tbl_session (id, timestamp)
        VALUES (?, NOW())
    ");
    
    error_log("create_session.php - Attempting to insert session for user_id: " . $user_id);
    
    $result = $stmt->execute([$user_id]);
    
    if (!$result) {
        $errorInfo = $stmt->errorInfo();
        $error_msg = "Failed to insert session: " . ($errorInfo[2] ?? "Unknown error");
        error_log("create_session.php - " . $error_msg);
        echo json_encode([
            "success" => false,
            "message" => $error_msg,
            "error_info" => $errorInfo
        ]);
        exit();
    }
    
    // Get the last insert ID - try both methods
    $session_id = $pdo->lastInsertId();
    
    // If lastInsertId() returns 0 or false, try alternative method
    if (!$session_id || $session_id == 0) {
        // Alternative: Query the database directly
        $check_stmt = $pdo->prepare("SELECT session_id FROM tbl_session WHERE id = ? ORDER BY session_id DESC LIMIT 1");
        $check_stmt->execute([$user_id]);
        $last_session = $check_stmt->fetch(PDO::FETCH_ASSOC);
        if ($last_session && isset($last_session['session_id'])) {
            $session_id = $last_session['session_id'];
            error_log("create_session.php - Retrieved session_id via SELECT: " . $session_id);
        } else {
            $error_msg = "Failed to get session_id after insert. lastInsertId returned: " . $session_id;
            error_log("create_session.php - " . $error_msg);
            echo json_encode([
                "success" => false,
                "message" => $error_msg
            ]);
            exit();
        }
    }
    
    error_log("create_session.php - Insert successful. session_id: " . $session_id);
    
    // Verify the session was actually created
    $verify_stmt = $pdo->prepare("SELECT session_id, id, timestamp FROM tbl_session WHERE session_id = ?");
    $verify_stmt->execute([$session_id]);
    $verified_session = $verify_stmt->fetch(PDO::FETCH_ASSOC);
    
    if (!$verified_session) {
        $error_msg = "Session was inserted but could not be verified. session_id: " . $session_id;
        error_log("create_session.php - " . $error_msg);
        echo json_encode([
            "success" => false,
            "message" => $error_msg
        ]);
        exit();
    }
    
    error_log("create_session.php - Session verified. session_id: " . $session_id . ", user_id: " . $verified_session['id']);
    
    echo json_encode([
        "success" => true,
        "session_id" => intval($session_id),
        "message" => "Session created successfully",
        "verified" => true
    ]);
    
} catch (PDOException $e) {
    $error_msg = "Database error: " . $e->getMessage();
    error_log("create_session.php - PDOException: " . $error_msg);
    error_log("create_session.php - Error code: " . $e->getCode());
    error_log("create_session.php - SQL State: " . $e->getCode());
    echo json_encode([
        "success" => false,
        "message" => $error_msg,
        "error_code" => $e->getCode(),
        "sql_state" => $e->errorInfo[0] ?? null
    ]);
} catch (Exception $e) {
    $error_msg = "General error: " . $e->getMessage();
    error_log("create_session.php - Exception: " . $error_msg);
    echo json_encode([
        "success" => false,
        "message" => $error_msg
    ]);
}
?>

