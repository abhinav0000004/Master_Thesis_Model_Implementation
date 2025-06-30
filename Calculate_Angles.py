import os
import numpy as np
import pandas as pd
import json

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle (in degrees) formed at point2 by the segments
    point2->point1 and point2->point3 in 3D space.
    """
    p1, p2, p3 = np.array(point1), np.array(point2), np.array(point3)
    v1, v2 = p1 - p2, p3 - p2
    
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return np.nan
    
    dot_product = np.clip(np.dot(v1 / norm_v1, v2 / norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def process_file(file_path, output_folder):
    # Extract filename details
    file_name = os.path.basename(file_path).replace('.npz', '')
    subject, action, variation, camera = file_name.split('_')[:4]
    camera_number = int(camera[1])  # Extract camera number
    
    # Load NPY data
    data = np.load(file_path)
    arr = data[data.files[0]]  # Assuming single array
    
    # Marker map
    marker_map = {
        'Hip': 0, 'RHip': 1, 'RKnee': 2, 'RFoot': 3, 'LHip': 4,
        'LKnee': 5, 'LFoot': 6, 'Spine': 7, 'Neck': 8, 'Nose': 9,
        'Head': 10, 'LShoulder': 11, 'LElbow': 12, 'LWrist': 13,
        'RShoulder': 14, 'RElbow': 15, 'RWrist': 16
    }
    
    # Angle triplets
    angle_triplets = {
        'l_elbow_angle':    ('LShoulder', 'LElbow', 'LWrist'),
        'r_elbow_angle':    ('RShoulder', 'RElbow', 'RWrist'),
        'l_shoulder_angle': ('Neck', 'LShoulder', 'LElbow'),
        'r_shoulder_angle': ('Neck', 'RShoulder', 'RElbow'),
        'l_hip_angle':      ('LShoulder', 'LHip', 'LKnee'),
        'r_hip_angle':      ('RShoulder', 'RHip', 'RKnee'),
        'l_knee_angle':     ('LFoot', 'LKnee', 'LHip'),
        'r_knee_angle':     ('RFoot', 'RKnee', 'RHip')
    }
    
    # Process frames
    num_frames = arr.shape[0]
    angles_list = []
    for frame_idx in range(num_frames):
        frame_angles = {
            'Subject': subject,
            'Action': action,
            'Variation': variation,
            'Camera': camera_number,
            'Frame': frame_idx
        }

        for angle_name, (mk1, mk2, mk3) in angle_triplets.items():
            try:
                pt1 = arr[frame_idx, marker_map[mk1], :]
                pt2 = arr[frame_idx, marker_map[mk2], :]
                pt3 = arr[frame_idx, marker_map[mk3], :]
                frame_angles[angle_name] = calculate_angle(pt1, pt2, pt3)
            except KeyError:
                frame_angles[angle_name] = np.nan
        
        angles_list.append(frame_angles)
    
    df = pd.DataFrame(angles_list)
    
    # Create the "Angles Processed" DataFrame
    if not df.empty:
        first_frame_angles = df.iloc[0].filter(like='_angle').to_numpy()
        processed_angles = df.iloc[1:].copy()
        for angle_column in df.filter(like='_angle').columns:
            processed_angles[angle_column] -= first_frame_angles[processed_angles.filter(like='_angle').columns.get_loc(angle_column)]
    
    # Save to Excel
    output_file = os.path.join(output_folder, f"{subject}_{action}_{variation}_C{camera_number}.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='Angles', index=False)
        processed_angles.to_excel(writer, sheet_name='Angles Processed', index=False)
    print(f"Processed and saved: {output_file}")

def process_all_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".npz"):
                file_path = os.path.join(root, file_name)
                process_file(file_path, output_folder)

def process_json_file(file_path, output_folder):
    with open(file_path, 'r') as file:
        data = json.load(file)

    file_name = os.path.basename(file_path)
    subject, action, variation = file_name.split('_')[:3]
    variation = variation.replace('.json', '')  # Remove the .json extension
    frames = {int(k): pd.DataFrame.from_dict(v, orient='index', columns=['x', 'y', 'z']) for k, v in data.items()}

    # Define all angle triplets
    angle_triplets = {
        'l_elbow_angle': ('l_shoulder', 'l_elbow', 'l_wrist'),
        'r_elbow_angle': ('r_shoulder', 'r_elbow', 'r_wrist'),
        'l_shoulder_angle': ('neck', 'l_shoulder', 'l_elbow'),
        'r_shoulder_angle': ('neck', 'r_shoulder', 'r_elbow'),
        'l_hip_angle': ('l_shoulder', 'l_hip', 'l_knee'),
        'r_hip_angle': ('r_shoulder', 'r_hip', 'r_knee'),
        'l_knee_angle': ('l_ankle', 'l_knee', 'l_hip'),
        'r_knee_angle': ('r_ankle', 'r_knee', 'r_hip'),
    }

    angles_list = []

    # Process the first 503 frames only
    for frame_number, frame in sorted(frames.items())[:503]:
        try:
            # Compute midpoints
            hip_l = (frame.loc['LASI'].to_numpy() + frame.loc['LPSI'].to_numpy()) / 2
            hip_r = (frame.loc['RASI'].to_numpy() + frame.loc['RPSI'].to_numpy()) / 2
            wrist_l = (frame.loc['LWRA'].to_numpy() + frame.loc['LWRB'].to_numpy()) / 2
            wrist_r = (frame.loc['RWRA'].to_numpy() + frame.loc['RWRB'].to_numpy()) / 2
        except KeyError as e:
            print(f"Missing marker(s) in frame {frame_number}: {e}")
            continue

        # Map keypoints for angle calculation
        relevant_keypoints = {
            'l_wrist': wrist_l,
            'r_wrist': wrist_r,
            'l_elbow': frame.loc['LELB'].to_numpy(),
            'r_elbow': frame.loc['RELB'].to_numpy(),
            'l_shoulder': frame.loc['LSHO'].to_numpy(),
            'r_shoulder': frame.loc['RSHO'].to_numpy(),
            'l_hip': hip_l,
            'r_hip': hip_r,
            'l_knee': frame.loc['LKNE'].to_numpy(),
            'r_knee': frame.loc['RKNE'].to_numpy(),
            'l_ankle': frame.loc['LANK'].to_numpy(),
            'r_ankle': frame.loc['RANK'].to_numpy(),
            'neck': frame.loc['C7'].to_numpy(),
        }

        # Calculate angles for this frame
        frame_angles = {'Subject': subject, 'Action': action, 'Variation': variation, 'Camera': 1, 'Frame': frame_number}
        for angle_name, (kp1, kp2, kp3) in angle_triplets.items():
            try:
                frame_angles[angle_name] = calculate_angle(relevant_keypoints[kp1], relevant_keypoints[kp2], relevant_keypoints[kp3])
            except KeyError:
                frame_angles[angle_name] = np.nan
        
        angles_list.append(frame_angles)

    # Convert to DataFrame
    df = pd.DataFrame(angles_list)

    # Duplicate for Camera=2
    if not df.empty:
        df_camera2 = df.copy()
        df_camera2['Camera'] = 2  # Set Camera to 2
        angles_final = pd.concat([df, df_camera2])

    # Create the "Angles Processed" DataFrame for Camera 1
    if not df.empty:
        first_frame_angles = df.iloc[0].filter(like='_angle').to_numpy()
        processed_angles = df.iloc[1:].copy()  # Exclude the first frame
        for angle_column in df.filter(like='_angle').columns:
            processed_angles[angle_column] -= first_frame_angles[processed_angles.filter(like='_angle').columns.get_loc(angle_column)]

        # Duplicate the processed data for Camera 2
        processed_angles_camera2 = processed_angles.copy()
        processed_angles_camera2['Camera'] = 2  # Set Camera to 2

        # Combine processed data for Camera 1 and Camera 2
        processed_final = pd.concat([processed_angles, processed_angles_camera2])

    # Save to Excel with two sheets
    output_file = os.path.join(output_folder, f"{subject}_{action}_{variation}.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        angles_final.to_excel(writer, sheet_name='Angles', index=False)  # Original angles
        processed_final.to_excel(writer, sheet_name='Angles Processed', index=False)  # Processed angles
    print(f"Processed and saved to Excel: {output_file}")

def process_all_json_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            process_json_file(os.path.join(input_folder, file_name), output_folder)

def consolidate_excel_files(input_folder, output_file):
    # Initialize output files for two sheets
    angles_output_file = output_file.replace('.xlsx', '_Angles.xlsx')
    processed_output_file = output_file.replace('.xlsx', '_Angles_Processed.xlsx')

    # Create Excel writers for each sheet
    with pd.ExcelWriter(angles_output_file, engine='openpyxl', mode='w') as angles_writer, \
         pd.ExcelWriter(processed_output_file, engine='openpyxl', mode='w') as processed_writer:

        first_angles = True  # Track if it's the first write for "Angles" sheet
        first_processed = True  # Track if it's the first write for "Angles Processed" sheet

        # Process each file in the input folder
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.xlsx'):  # Only process Excel files
                file_path = os.path.join(input_folder, file_name)
                
                # Skip files starting with specific names
                if file_name.startswith("S3_jumpingjacks_normal_C1") or file_name.startswith("S3_jumpingjacks_normal_C2"):
                    print(f"Skipping file: {file_name}")
                    continue

                try:
                    # Read the two sheets from the file
                    angles_df = pd.read_excel(file_path, sheet_name='Angles')
                    processed_df = pd.read_excel(file_path, sheet_name='Angles Processed')

                    # Append to the respective output files
                    if first_angles:
                        angles_df.to_excel(angles_writer, index=False, header=True)
                        first_angles = False
                    else:
                        angles_df.to_excel(angles_writer, index=False, header=False, startrow=angles_writer.sheets['Sheet1'].max_row)

                    if first_processed:
                        processed_df.to_excel(processed_writer, index=False, header=True)
                        first_processed = False
                    else:
                        processed_df.to_excel(processed_writer, index=False, header=False, startrow=processed_writer.sheets['Sheet1'].max_row)

                    print(f"Consolidated file: {file_name}")

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    print(f"Consolidation complete! Files saved as:\n{angles_output_file}\n{processed_output_file}")

# Paths
model_input_folder = r"Your\Folder\Path"  # This will recursively read all .npz files in subfolders
model_output_excel_folder = r"Data\Model Calculated Angles"  # Folder to save processed Excel files
model_output_file = 'Data\Model Calculated Angles\Model.xlsx'  # Base name for Model output files
groundtruth_input_json_folder = r"Data\Ground Truth" 
groundtruth_output_excel_folder = r"Data\Ground Truth Calculated Angles" 
groundtruth_output_file = r"Data\Ground Truth Calculated Angles\Ground_Truth.xlsx"   # Base name for GroundTruth output files

# Process all NPY files
process_all_files(model_input_folder, model_output_excel_folder)

# Consolidate all Excel files
consolidate_excel_files(model_output_excel_folder, model_output_file)

# Process all files
process_all_json_files(groundtruth_input_json_folder, groundtruth_output_excel_folder)

# Consolidate all Excel files
consolidate_excel_files(groundtruth_output_excel_folder, groundtruth_output_file)