from app import app


# UPDATING COMPONENTS
import update_callbacks


# function for CHARTING imported in save_plot_callback

# CREATE EXCEL TABLE
import excel_callback


# GENERATE CLIENT BULLETIN
import bulletin_callback


# PRESET PROCESS
import preset_callback

# DATA UPLOAD PROCESS
import data_upload_callback


# CALIBRATION START AND STOP
import calibration_callbacks


# SAVE PLOT
import save_plot_callback


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8050)
