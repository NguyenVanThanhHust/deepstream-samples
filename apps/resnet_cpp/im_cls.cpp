#include <gst/gst.h>
#include <glib.h>

#include <memory>
#include <cassert>

#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf=(GstBuffer*)info->data;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop*) data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
        g_print("End of stream \n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR:
        {
            gchar* debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("Error from element %s: %s \n", 
                GST_OBJECT_NAME(msg->src), error->message
            );
            if (debug)
                g_printerr("Error details: %s \n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
    default:
        break;
    }
    return true;
}


int main(int argc, char *argv[]) {
    GMainLoop *loop=NULL;
    GstElement *pipeline=NULL, *source=NULL, *h264parser=NULL, *decoder=NULL, 
    *streammux=NULL, *pgie=NULL, *nvvidconv=NULL, *nvosd=NULL, *sink=NULL;

    int current_device=-1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
    gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");
    
    GstBus *bus;
    guint bus_watch_id = 0;
    GstPad *osd_sink_pad=NULL;

    // Stanard Gstreamer initialization
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, false);

    // Create gstreamers elements
    // Create Pipeline elemetsn to connect other elements
    pipeline = gst_pipeline_new("im-cls-pipeline");
    if(!pipeline) {
        g_printerr("Can't create pipeline. Exiting \n");
        return -1;
    }

    // Create element filesrc to read from file 
    source = gst_element_factory_make("filesrc", "file-source");
    if(!source) {
        g_printerr("Can't create filesrc to read h264 file. Exiting \n");
        return -1;
    }

    // Input file is a single h264 encoded file
    // So we needs to h264parser
    // Create element h264parse to parse h264 stream
    h264parser = gst_element_factory_make("h264parse", "h264-parser");
    if(!source) {
        g_printerr("Can't create h264parse to parse h264 file. Exiting \n");
        return -1;
    }

    // use nvdec_h264 for hardware accelerated decode on GPU
    decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
    if(!decoder) {
        g_printerr("Can't create Nv v4l2 decoder. Exiting \n");
        return -1;
    }

    // Create nvstreammux instance to form batches from one or more sources.
    streammux = gst_element_factory_make("nvstreammux", "Stream-muxer");
    if(!streammux) {
        g_printerr("Unable to create NvStreamMux. Exiting \n");
        return -1;
    }

    // Use nvinfer to run inferencing on decoder's output,
    // behaviour of inferencing is set through config file
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    if(!pgie) {
        g_printerr("Unable to create pgie with nvinfer. Exiting \n");
        return -1;
    }

    // Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    if(!nvvidconv) {
        g_printerr("Unable to create nvvidconvr. Exiting \n");
        return -1;
    }

    // Create OSD to draw on the converted RGBA buffer
    nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    if(!nvosd) {
        g_printerr("Unable to create nvosd. Exiting \n");
        return -1;
    }

    // Finally render the osd output
    if(prop.integrated) {
        sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
        if(!sink) {
            g_printerr("Unable to create nv3dsink. Exiting \n");
            return -1;
        }
    } 
    else {
    #ifdef __aarch64__
        sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
    #else
        sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
        if(!sink) {
            g_printerr("Unable to create eglsink. Exiting \n");
            return -1;
        }
    #endif
    }
    
    // Settting properties of elements here
    // set h264 file to run
    g_object_set(G_OBJECT(source), "location", argv[1], NULL);

    g_object_set(G_OBJECT(streammux), "batch-size", 1, NULL);
    g_object_set(G_OBJECT(streammux), "width", 1920, NULL);
    g_object_set(G_OBJECT(streammux), "height", 1080, NULL);
    g_object_set(G_OBJECT(streammux), "batched-push-timeout", 100000, NULL);

    g_object_set(G_OBJECT(pgie), "config-file-path", "nvinfer_config.txt", NULL);

    // Message handler
    bus = gst_pipeline_get_bus (GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // setup pipeline
    // We can connect elements one by one as in python version
    // but c version allow us to connect multiple element at once too
    gst_bin_add_many(GST_BIN(pipeline),
      source, h264parser, decoder, streammux, pgie,
      nvvidconv, nvosd, sink, NULL);

    // Pad to link decoder and streammux
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    // Link elements together, decoder and streammux are linked by pad in previous step
    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }
    
    if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }

    // create pad to parse output of ods 
    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
            osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);

    /* Set the pipeline to "playing" state */
    g_print ("Using file: %s\n", argv[1]);
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);

    return 0;
}