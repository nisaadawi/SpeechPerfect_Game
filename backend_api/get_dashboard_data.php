<?php
/**
 * Get dashboard data for a user
 * Returns all sessions with their complete data from tbl_session
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
$user_id = intval($input['user_id'] ?? 0);

if (!$user_id) {
    echo json_encode(["success" => false, "message" => "Missing user_id"]);
    exit();
}

try {
    // Fetch all sessions for this user from master table
    $sessions_stmt = $pdo->prepare("
        SELECT 
            session_id,
            id as user_id,
            timestamp,
            attention_percentage,
            attention_cat_label,
            audio_file,
            filler_per_min,
            filler_cat_label,
            avg_hr,
            hr_cat_label,
            mfcc_std_dev,
            mfcc_cat_label,
            avg_pause,
            pause_cat_label,
            stress_probability,
            stress_cat_label,
            wpm,
            wpm_cat_label,
            speech_param,
            speech_score,
            speech_label
        FROM tbl_session 
        WHERE id = ? 
        ORDER BY timestamp DESC
    ");
    $sessions_stmt->execute([$user_id]);
    $sessions = $sessions_stmt->fetchAll(PDO::FETCH_ASSOC);
    
    // Fetch latest HR and Attention for summary (optional - can also get from latest session)
    $hr_stmt = $pdo->prepare("
        SELECT hr_min, hr_max, hr_avg, hr_status, timestamp
        FROM tbl_heart_rate 
        WHERE id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    ");
    $hr_stmt->execute([$user_id]);
    $hr_data = $hr_stmt->fetch(PDO::FETCH_ASSOC);
    
    $att_stmt = $pdo->prepare("
        SELECT focus_percentage, attention_status, timestamp
        FROM tbl_attention 
        WHERE id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    ");
    $att_stmt->execute([$user_id]);
    $att_data = $att_stmt->fetch(PDO::FETCH_ASSOC);
    
    // Format sessions for backward compatibility with existing Dashboard code
    $speech_results = [];
    $wpm_data = [];
    $filler_data = [];
    $pause_data = [];
    $mfcc_data = [];
    $stress_data = [];
    
    foreach ($sessions as $session) {
        $speech_results[] = [
            'result_id' => $session['session_id'],
            'speech_param' => $session['speech_param'] ?? null,
            'speech_score' => $session['speech_score'] ?? null,
            'speech_label' => $session['speech_label'] ?? null,
            'timestamp' => $session['timestamp']
        ];
        
        $wpm_data[] = [
            'wpm_id' => $session['session_id'],
            'word_per_min' => $session['wpm'],
            'wpm_label' => $session['wpm_cat_label'] >= 1.0 ? 'Good' : ($session['wpm_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
            'timestamp' => $session['timestamp']
        ];
        
        $filler_data[] = [
            'filler_id' => $session['session_id'],
            'filler_per_min' => $session['filler_per_min'],
            'filler_label' => $session['filler_cat_label'] >= 1.0 ? 'Good' : ($session['filler_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
            'timestamp' => $session['timestamp']
        ];
        
        $pause_data[] = [
            'pause_id' => $session['session_id'],
            'avg_pause_sec' => $session['avg_pause'],
            'pause_label' => $session['pause_cat_label'] >= 1.0 ? 'Good' : ($session['pause_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
            'timestamp' => $session['timestamp']
        ];
        
        $mfcc_data[] = [
            'mfcc_id' => $session['session_id'],
            'mfcc_std_dev' => $session['mfcc_std_dev'],
            'mfcc_label' => $session['mfcc_cat_label'] >= 1.0 ? 'Good' : ($session['mfcc_cat_label'] >= 0.5 ? 'Moderate' : 'Severe'),
            'timestamp' => $session['timestamp']
        ];
        
        $stress_data[] = [
            'stress_id' => $session['session_id'],
            'stress_probability' => $session['stress_probability'],
            'stress_label' => $session['stress_cat_label'] >= 1.0 ? 'Stressed' : 'Not Stressed',
            'timestamp' => $session['timestamp']
        ];
    }
    
    echo json_encode([
        "success" => true,
        "sessions" => $sessions,  // New: Complete session data
        "speech_results" => $speech_results,  // For backward compatibility
        "wpm_data" => $wpm_data,
        "filler_data" => $filler_data,
        "pause_data" => $pause_data,
        "mfcc_data" => $mfcc_data,
        "stress_data" => $stress_data,
        "heart_rate" => $hr_data,
        "attention" => $att_data
    ]);
    
} catch (PDOException $e) {
    echo json_encode([
        "success" => false,
        "message" => "Database error: " . $e->getMessage()
    ]);
}
?>
