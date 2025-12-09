<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

require_once 'db_config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["success" => false, "message" => "Only POST method allowed"]);
    exit();
}

// Get JSON input
$input = json_decode(file_get_contents('php://input'), true);

if (!$input) {
    echo json_encode(["success" => false, "message" => "Invalid JSON input"]);
    exit();
}

// Validate required fields
$required_fields = ['user_id', 'eye_tracker', 'heart_rate'];
foreach ($required_fields as $field) {
    if (!isset($input[$field])) {
        echo json_encode(["success" => false, "message" => "Missing required field: $field"]);
        exit();
    }
}

$user_id = intval($input['user_id']);
$eye_tracker = $input['eye_tracker'];
$heart_rate = $input['heart_rate'];
$audio_file = $input['audio_file'] ?? null;

try {
    $pdo->beginTransaction();
    
    $errors = [];
    
    // 1. Save attention data (tbl_attention)
    if (isset($eye_tracker['focus_percentage']) && isset($eye_tracker['attention_status'])) {
        $focus_percentage = intval($eye_tracker['focus_percentage']);
        $attention_status = $eye_tracker['attention_status'];
        
        // Determine attention status if not provided
        if (empty($attention_status)) {
            if ($focus_percentage >= 90) {
                $attention_status = "Good";
            } elseif ($focus_percentage >= 70) {
                $attention_status = "Fair";
            } else {
                $attention_status = "Poor";
            }
        }
        
        $stmt = $pdo->prepare("INSERT INTO tbl_attention (id, focus_percentage, attention_status) VALUES (?, ?, ?)");
        $stmt->execute([$user_id, $focus_percentage, $attention_status]);
    } else {
        $errors[] = "Missing eye tracker data (focus_percentage or attention_status)";
    }
    
    // 2. Save heart rate data (tbl_heart_rate)
    if (isset($heart_rate['min_bpm']) && isset($heart_rate['max_bpm']) && isset($heart_rate['avg_bpm']) && isset($heart_rate['hr_status'])) {
        $hr_min = intval($heart_rate['min_bpm']);
        $hr_max = intval($heart_rate['max_bpm']);
        $hr_avg = intval($heart_rate['avg_bpm']);
        $hr_status = $heart_rate['hr_status'];
        
        $stmt = $pdo->prepare("INSERT INTO tbl_heart_rate (id, hr_min, hr_max, hr_avg, hr_status) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute([$user_id, $hr_min, $hr_max, $hr_avg, $hr_status]);
    } else {
        $errors[] = "Missing heart rate data (min_bpm, max_bpm, avg_bpm, or hr_status)";
    }
    
    // 3. Save audio file path (tbl_audio)
    if ($audio_file) {
        // Store relative path or filename only
        // If full path is provided, extract just the filename
        // Example: "recordings/speech_123456.wav" -> "speech_123456.wav"
        $audio_filename = basename($audio_file);
        
        // If you want to store the full relative path, use:
        // $audio_filename = $audio_file;
        
        $stmt = $pdo->prepare("INSERT INTO tbl_audio (id, audio_file) VALUES (?, ?)");
        $stmt->execute([$user_id, $audio_filename]);
    } else {
        // Audio file is optional - don't treat as error
        // $errors[] = "Missing audio file path";
    }
    
    if (!empty($errors)) {
        $pdo->rollBack();
        echo json_encode([
            "success" => false,
            "message" => "Some data could not be saved",
            "errors" => $errors
        ]);
        exit();
    }
    
    $pdo->commit();
    
    echo json_encode([
        "success" => true,
        "message" => "Data saved successfully",
        "user_id" => $user_id
    ]);
    
} catch (PDOException $e) {
    $pdo->rollBack();
    echo json_encode([
        "success" => false,
        "message" => "Database error: " . $e->getMessage()
    ]);
} catch (Exception $e) {
    $pdo->rollBack();
    echo json_encode([
        "success" => false,
        "message" => "Error: " . $e->getMessage()
    ]);
}
?>

