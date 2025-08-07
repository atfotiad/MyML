package com.atfotiad.myml

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.atfotiad.myml.databinding.ActivityMainBinding
import com.atfotiad.myml.ml.MobilenetV2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var mainBinding: ActivityMainBinding
    private lateinit var bitmap: Bitmap
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(mainBinding.root)


        val fileName = "labels2.txt"
        val inputString = application.assets.open(fileName).bufferedReader().readText()
        val townList = inputString.split("\n")

        mainBinding.open.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, 100)


        }
        mainBinding.process.setOnClickListener {
            val resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 300, 300, true)
            val width = resized.width
            val height = resized.height
            Log.d("", "onCreate: resized: $width $height ${resized}")
            
            //val model = MobilenetV110224Quant.newInstance(this)
            //val model =  Test.newInstance(this)
            val model = MobilenetV2.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 =
                TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 3), DataType.FLOAT32)

            Log.d("", "onCreate: inputBuffer: ${inputFeature0.buffer}")
            val tensorBuffer = TensorImage.fromBitmap(resized.copy(Bitmap.Config.ARGB_8888,true))
            val byteBuffer = tensorBuffer.buffer
            Log.d("", "onCreate: byteBuffer: $byteBuffer")

            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.

            //val image = TensorImage.fromBitmap(resized.copy(Bitmap.Config.ARGB_8888,true))

// Runs model inference and gets result.
           /* val outputs = model.process(image)
            val probability = outputs.probabilityAsCategoryList.apply {
                sortByDescending {
                    it.score
                }
            }.first()*/

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val max = getMaxIndex(outputFeature0.floatArray)


            mainBinding.recognized.text = buildString {
                append(outputFeature0)
                append("-")
                append(townList[max])
       /* append(probability.label)
        append(" _ ")
        append(probability.score)*/
    }
                    //townList[max]

// Releases model resources if no longer used.
            model.close()

        }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        mainBinding.imageView.setImageURI(data?.data)
        if (data!= null) {
            val uri: Uri = data.data!!

            bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri))
        }

    }

    private fun getMaxIndex(arr: FloatArray): Int {
        var index = 0
        var min = 0.0f
        for (i in 0..4) {
            if (arr[i] > min) {
                index = i
                min = arr[i]
            }
        }

        return index
    }

}