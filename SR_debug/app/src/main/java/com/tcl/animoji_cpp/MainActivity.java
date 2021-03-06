package com.tcl.animoji_cpp;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
//        TextView view = new TextView(this);
//        view.setText(String.valueOf(FacePredictionFromJNI()));
        setContentView(R.layout.activity_main);
        TextView buildView= (TextView) this.findViewById(R.id.build_time);
        buildView.setText(String.valueOf(mainFromJNI()));
        TextView detectView = (TextView) this.findViewById(R.id.detect_time);
        detectView.setText(String.valueOf("1111"));
        TextView landmarkView = (TextView) this.findViewById(R.id.landmark_time);
        landmarkView.setText(String.valueOf("2222"));
//        setContentView(view);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
//    public native int resultFromJNI();
    public native int mainFromJNI();

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };


    public static void verifyStoragePermissions(Activity activity) {

        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
