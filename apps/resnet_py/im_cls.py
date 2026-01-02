import os
import sys
import argparse
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call

import pyds
from resnet_parser import nvds_infer_parse_custom_resnet

MUXER_BATCH_TIMEOUT_USEC = 33000

def pgie_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the GstBuffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_user_meta = frame_meta.frame_user_meta_list
        frame_number = frame_meta.frame_num
        output_class = "class_fake"
        while l_user_meta is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
            except StopIteration:
                break
            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                # Cast user_meta_data to NvDsInferTensorMeta
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                output_class = nvds_infer_parse_custom_resnet(tensor_meta)
            try:
                l_user_meta = l_user_meta.next
            except StopIteration:
                break
        # Get or create NvDsDisplayMeta for the current frame
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if display_meta is None:
            print("Failed to acquire NvDsDisplayMeta from pool. Skipping display.")
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        # Create NvOSD_TextParams for your string
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta) #Retrieve NvDsDisplayMeta object from given NvDsBatchMeta object. See NvDsMeta docs for more details.
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0] #Retrieve NvOSD_TextParams object from list in display meta. 
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = f"Frame Number={frame_number} Class={output_class}"

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif" #Set attributes of our NvOSD_TextParams object's NvOSD_FontParams member 
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0) #See NvOSD_ColorParams for more details.

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1 #Set boolean indicating that text has bg color to true. 
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta) #Add display meta to frame after setting text params attributes.

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	


def main(args):
    pgie_type = args.pgie_type
    input_video = args.input_video
    
    platform_info = PlatformInfo()
    # Init Gstreamer
    Gst.init(None)

    # Create gstreamer pipeline
    print("Creating pipeline")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Can not create pipeline")

    # Create element to read from file
    print("Create source")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write("Unable to create file source")

    # Create elements to form pipeline in gstreamer
    # Input file is a single h264 encoded file
    # So we needs to h264parser
    print("Create H264 Parser")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write("Unable to create h264 parser")
    
    # use nvdec_h264 for hardware accelerated decode on GPU
    print("Create decoder")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write("Unable to create Nv v4l2 decoder")
    
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    if pgie_type == "nvinfer":
        # Use nvinfer to run inferencing on decoder's output,
        # behaviour of inferencing is set through config file
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie with nvinfer \n")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie with nvinfer-server \n")
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if platform_info.is_integrated_gpu():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    print(f"Playing file {input_video}")
    source.set_property('location', input_video)
    if os.environ.get('USE_NEW_NVSTREAMMUX') != 'yes': # Only set these properties if not using new gst-nvstreammux
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    streammux.set_property('batch-size', 1)
    if pgie_type == "nvinfer":
        pgie.set_property('config-file-path', "nvinfer_config.txt")
    else:
        pgie.set_property('config-file-path', "nvinfer_server_config.txt")

    print("Adding element to Pipeline ")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Add a probe on the primary-infer soruce pad to get infer output tensors
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad of primary infer engine")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

def get_args():
    parser = argparse.ArgumentParser(prog="sample resnet")
    parser.add_argument("--pgie_type", type=str, choices=["nvinfer", "nvinfer-server"], default="nvinfer")
    parser.add_argument("--input_video", type=str, default="/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    sys.exit(main(args))