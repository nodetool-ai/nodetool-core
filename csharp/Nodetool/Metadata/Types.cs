using MessagePack;
using System.Collections.Generic;
namespace Nodetool.Metadata {
public enum DateCriteria
{
    BEFORE,
    SINCE,
    ON,
}

public enum EmailFlag
{
    SEEN,
    UNSEEN,
    ANSWERED,
    UNANSWERED,
    FLAGGED,
    UNFLAGGED,
}

public enum OpenAIEmbeddingModel
{
    ADA_002,
    SMALL,
    LARGE,
}

public enum Provider
{
    AIME,
    OpenAI,
    Anthropic,
    Replicate,
    HuggingFace,
    Ollama,
    Comfy,
    Local,
    Gemini,
    Empty,
}

public enum SeabornEstimator
{
    MEAN,
    MEDIAN,
    COUNT,
    SUM,
    MIN,
    MAX,
    VAR,
    STD,
}

public enum SeabornPlotType
{
    SCATTER,
    LINE,
    RELPLOT,
    HISTPLOT,
    KDEPLOT,
    ECDFPLOT,
    RUGPLOT,
    DISTPLOT,
    STRIPPLOT,
    SWARMPLOT,
    BOXPLOT,
    VIOLINPLOT,
    BOXENPLOT,
    POINTPLOT,
    BARPLOT,
    COUNTPLOT,
    REGPLOT,
    LMPLOT,
    RESIDPLOT,
    HEATMAP,
    CLUSTERMAP,
    JOINTPLOT,
    PAIRPLOT,
    FACETGRID,
}

public enum SeabornStatistic
{
    COUNT,
    FREQUENCY,
    PROBABILITY,
    PERCENT,
    DENSITY,
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class AssetRef
{
    public string type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class AudioChunk
{
    public object type { get; set; }
    public List<double> timestamp { get; set; }
    public string text { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class AudioRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class BaseType
{
    public string type { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class BoundingBox
{
    public object type { get; set; }
    public double xmin { get; set; }
    public double ymin { get; set; }
    public double xmax { get; set; }
    public double ymax { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CLIP
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CLIPFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CLIPVision
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CLIPVisionFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CLIPVisionOutput
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CalendarEvent
{
    public object type { get; set; }
    public string title { get; set; }
    public Datetime start_date { get; set; }
    public Datetime end_date { get; set; }
    public string calendar { get; set; }
    public string location { get; set; }
    public string notes { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ChartConfig
{
    public object type { get; set; }
    public string title { get; set; }
    public string x_label { get; set; }
    public string y_label { get; set; }
    public bool legend { get; set; }
    public ChartData data { get; set; }
    public object height { get; set; }
    public object aspect { get; set; }
    public object x_lim { get; set; }
    public object y_lim { get; set; }
    public object? x_scale { get; set; }
    public object? y_scale { get; set; }
    public object legend_position { get; set; }
    public object palette { get; set; }
    public object hue_order { get; set; }
    public object hue_norm { get; set; }
    public object sizes { get; set; }
    public object size_order { get; set; }
    public object size_norm { get; set; }
    public object marginal_kws { get; set; }
    public object joint_kws { get; set; }
    public object? diag_kind { get; set; }
    public bool corner { get; set; }
    public object center { get; set; }
    public object vmin { get; set; }
    public object vmax { get; set; }
    public object cmap { get; set; }
    public bool annot { get; set; }
    public string fmt { get; set; }
    public bool square { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ChartConfigSchema
{
    public string title { get; set; }
    public string x_label { get; set; }
    public string y_label { get; set; }
    public bool legend { get; set; }
    public ChartDataSchema data { get; set; }
    public double? height { get; set; }
    public double? aspect { get; set; }
    public List<double>? x_lim { get; set; }
    public List<double>? y_lim { get; set; }
    public object? x_scale { get; set; }
    public object? y_scale { get; set; }
    public object? legend_position { get; set; }
    public string? palette { get; set; }
    public List<string>? hue_order { get; set; }
    public List<double>? hue_norm { get; set; }
    public List<double>? sizes { get; set; }
    public List<string>? size_order { get; set; }
    public List<double>? size_norm { get; set; }
    public object? marginal_kws { get; set; }
    public object? joint_kws { get; set; }
    public object? diag_kind { get; set; }
    public bool corner { get; set; }
    public double? center { get; set; }
    public double? vmin { get; set; }
    public double? vmax { get; set; }
    public string? cmap { get; set; }
    public bool annot { get; set; }
    public string fmt { get; set; }
    public bool square { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ChartData
{
    public object type { get; set; }
    public List<DataSeries> series { get; set; }
    public object row { get; set; }
    public object col { get; set; }
    public object col_wrap { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ChartDataSchema
{
    public List<DataSeriesSchema> series { get; set; }
    public string? row { get; set; }
    public string? col { get; set; }
    public int? col_wrap { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ChatConversation
{
    public List<string> messages { get; set; }
    public string response { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class CheckpointFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Collection
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ColorRef
{
    public object type { get; set; }
    public object value { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ColumnDef
{
    public string name { get; set; }
    public object data_type { get; set; }
    public string description { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ComfyData
{
    public string type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ComfyModel
{
    public string type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Conditioning
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ControlNet
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ControlNetFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class DataSeries
{
    public object type { get; set; }
    public string name { get; set; }
    public string x { get; set; }
    public object y { get; set; }
    public object hue { get; set; }
    public object size { get; set; }
    public object style { get; set; }
    public object weight { get; set; }
    public object color { get; set; }
    public SeabornPlotType plot_type { get; set; }
    public object estimator { get; set; }
    public object ci { get; set; }
    public int n_boot { get; set; }
    public object units { get; set; }
    public object seed { get; set; }
    public object stat { get; set; }
    public object bins { get; set; }
    public object binwidth { get; set; }
    public object binrange { get; set; }
    public object discrete { get; set; }
    public string line_style { get; set; }
    public string marker { get; set; }
    public double alpha { get; set; }
    public object? orient { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class DataSeriesSchema
{
    public string name { get; set; }
    public string x { get; set; }
    public string? y { get; set; }
    public string? hue { get; set; }
    public string? size { get; set; }
    public string? style { get; set; }
    public string? weight { get; set; }
    public string? color { get; set; }
    public SeabornPlotType plot_type { get; set; }
    public SeabornEstimator? estimator { get; set; }
    public double? ci { get; set; }
    public int n_boot { get; set; }
    public string? units { get; set; }
    public int? seed { get; set; }
    public SeabornStatistic? stat { get; set; }
    public int? bins { get; set; }
    public double? binwidth { get; set; }
    public List<double>? binrange { get; set; }
    public bool? discrete { get; set; }
    public string line_style { get; set; }
    public string marker { get; set; }
    public double alpha { get; set; }
    public object? orient { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class DataframeRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
    public object columns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Dataset
{
    public DataframeRef data { get; set; }
    public DataframeRef target { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Date
{
    public object type { get; set; }
    public int year { get; set; }
    public int month { get; set; }
    public int day { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class DateSearchCondition
{
    public object type { get; set; }
    public DateCriteria criteria { get; set; }
    public Datetime date { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Datetime
{
    public object type { get; set; }
    public int year { get; set; }
    public int month { get; set; }
    public int day { get; set; }
    public int hour { get; set; }
    public int minute { get; set; }
    public int second { get; set; }
    public int microsecond { get; set; }
    public string tzinfo { get; set; }
    public double utc_offset { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class DocumentRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Email
{
    public object type { get; set; }
    public string id { get; set; }
    public string sender { get; set; }
    public string subject { get; set; }
    public Datetime date { get; set; }
    public object body { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class EmailSearchCriteria
{
    public object type { get; set; }
    public string? from_address { get; set; }
    public string? to_address { get; set; }
    public string? subject { get; set; }
    public string? body { get; set; }
    public string? cc { get; set; }
    public string? bcc { get; set; }
    public DateSearchCondition? date_condition { get; set; }
    public List<EmailFlag> flags { get; set; }
    public List<string> keywords { get; set; }
    public string? folder { get; set; }
    public string? text { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Embeds
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Event
{
    public object type { get; set; }
    public string name { get; set; }
    public Dictionary<string, object> payload { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ExcelRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FaceAnalysis
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FaceEmbeds
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FilePath
{
    public object type { get; set; }
    public string path { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FolderPath
{
    public object type { get; set; }
    public string path { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FolderRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FontRef
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class FunctionDefinition
{
    public string name { get; set; }
    public string description { get; set; }
    public object parameters { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class GLIGEN
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class GLIGENFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Guider
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFAudioClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFAudioToAudio
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFAutomaticSpeechRecognition
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFCLIP
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFCLIPVision
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFCheckpointModel
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFComputerVision
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFControlNet
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFControlNetSDXL
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFDepthEstimation
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFDocumentQuestionAnswering
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFFeatureExtraction
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFFillMask
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFFlux
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFGOTOCR
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFIPAdapter
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageFeatureExtraction
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageSegmentation
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageTextToText
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageTo3D
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageToImage
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageToText
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFImageToVideo
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFLTXV
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFLoraSD
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFLoraSDConfig
{
    public object type { get; set; }
    public HFLoraSD lora { get; set; }
    public double strength { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFLoraSDXL
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFLoraSDXLConfig
{
    public object type { get; set; }
    public HFLoraSDXL lora { get; set; }
    public double strength { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFMaskGeneration
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFMiniCPM
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFNaturalLanguageProcessing
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFObjectDetection
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFQuestionAnswering
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFRealESRGAN
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFReranker
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFSentenceSimilarity
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFStableDiffusion
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFStableDiffusion3
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFStableDiffusionUpscale
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFStableDiffusionXL
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFStableDiffusionXLTurbo
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFStyleModel
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFSummarization
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTableQuestionAnswering
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFText2TextGeneration
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextGeneration
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextTo3D
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextToAudio
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextToImage
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextToSpeech
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTextToVideo
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTokenClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFTranslation
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFUnconditionalImageGeneration
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFUnet
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFVAE
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFVideoClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFVideoTextToText
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFVisualQuestionAnswering
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFVoiceActivityDetection
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFZeroShotAudioClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFZeroShotClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFZeroShotImageClassification
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HFZeroShotObjectDetection
{
    public object type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class HuggingFaceModel
{
    public string type { get; set; }
    public string repo_id { get; set; }
    public object path { get; set; }
    public object allow_patterns { get; set; }
    public object ignore_patterns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class IMAPConnection
{
    public object type { get; set; }
    public string host { get; set; }
    public int port { get; set; }
    public string username { get; set; }
    public string password { get; set; }
    public bool use_ssl { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class IPAdapter
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class IPAdapterFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ImageRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ImageSegmentationResult
{
    public object type { get; set; }
    public string label { get; set; }
    public ImageRef mask { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ImageTensor
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class InstantID
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class InstantIDFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class JSONRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class LORA
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class LORAFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class LanguageModel
{
    public object type { get; set; }
    public Provider provider { get; set; }
    public string id { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Latent
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class LlamaModel
{
    public object type { get; set; }
    public string name { get; set; }
    public string repo_id { get; set; }
    public string modified_at { get; set; }
    public int size { get; set; }
    public string digest { get; set; }
    public object details { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class LoRAConfig
{
    public object type { get; set; }
    public LORAFile lora { get; set; }
    public double strength { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class LoraWeight
{
    public object type { get; set; }
    public string url { get; set; }
    public double scale { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Mask
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Message
{
    public string type { get; set; }
    public object id { get; set; }
    public object workflow_id { get; set; }
    public object graph { get; set; }
    public object thread_id { get; set; }
    public object tools { get; set; }
    public object tool_call_id { get; set; }
    public string role { get; set; }
    public object name { get; set; }
    public object content { get; set; }
    public object tool_calls { get; set; }
    public object input_files { get; set; }
    public object output_files { get; set; }
    public object created_at { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class MessageAudioContent
{
    public object type { get; set; }
    public AudioRef audio { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class MessageDocumentContent
{
    public object type { get; set; }
    public DocumentRef document { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class MessageFile
{
    public object type { get; set; }
    public byte[] content { get; set; }
    public string mime_type { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class MessageImageContent
{
    public object type { get; set; }
    public ImageRef image { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class MessageTextContent
{
    public object type { get; set; }
    public string text { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class MessageVideoContent
{
    public object type { get; set; }
    public VideoRef video { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ModelFile
{
    public string type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ModelRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class NPArray
{
    public object type { get; set; }
    public object value { get; set; }
    public string dtype { get; set; }
    public List<int> shape { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class NodeRef
{
    public object type { get; set; }
    public string id { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Noise
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class OCRResult
{
    public object type { get; set; }
    public string text { get; set; }
    public double score { get; set; }
    public List<int> top_left { get; set; }
    public List<int> top_right { get; set; }
    public List<int> bottom_right { get; set; }
    public List<int> bottom_left { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ObjectDetectionResult
{
    public object type { get; set; }
    public string label { get; set; }
    public double score { get; set; }
    public BoundingBox box { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class OpenAIModel
{
    public object type { get; set; }
    public string id { get; set; }
    public string object { get; set; }
    public int created { get; set; }
    public string owned_by { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class OutputSlot
{
    public TypeMetadata type { get; set; }
    public string name { get; set; }
    public bool stream { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class OutputType
{
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class PlotlyConfig
{
    public object type { get; set; }
    public Dictionary<string, object> config { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class PlotlySeries
{
    public object type { get; set; }
    public string name { get; set; }
    public string x { get; set; }
    public object y { get; set; }
    public object color { get; set; }
    public object size { get; set; }
    public object symbol { get; set; }
    public object line_dash { get; set; }
    public string chart_type { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class REMBGSession
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class RSSEntry
{
    public object type { get; set; }
    public string title { get; set; }
    public string link { get; set; }
    public Datetime published { get; set; }
    public string summary { get; set; }
    public string author { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class RankingResult
{
    public object type { get; set; }
    public double score { get; set; }
    public string text { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class RecordType
{
    public object type { get; set; }
    public List<ColumnDef> columns { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class SKLearnModel
{
    public object type { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class SVGElement
{
    public object type { get; set; }
    public string name { get; set; }
    public Dictionary<string, string> attributes { get; set; }
    public object content { get; set; }
    public List<SVGElement> children { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class SVGRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Sampler
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Sigmas
{
    public object type { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class StatsModelsModel
{
    public object type { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class StyleModel
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class StyleModelFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class SubTask
{
    public object type { get; set; }
    public string id { get; set; }
    public object model { get; set; }
    public string content { get; set; }
    public string output_file { get; set; }
    public int max_iterations { get; set; }
    public object batch_processing { get; set; }
    public bool completed { get; set; }
    public int start_time { get; set; }
    public int end_time { get; set; }
    public List<string> input_files { get; set; }
    public string output_type { get; set; }
    public string output_schema { get; set; }
    public bool is_intermediate_result { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class Task
{
    public object type { get; set; }
    public string title { get; set; }
    public string description { get; set; }
    public List<SubTask> subtasks { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class TaskPlan
{
    public object type { get; set; }
    public string title { get; set; }
    public List<Task> tasks { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class TextChunk
{
    public object type { get; set; }
    public string text { get; set; }
    public string source_id { get; set; }
    public int start_index { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class TextRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ToolCall
{
    public string id { get; set; }
    public string name { get; set; }
    public Dictionary<string, object> args { get; set; }
    public object result { get; set; }
    public object subtask_id { get; set; }
    public object message { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class ToolName
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class UNet
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class UNetFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class UpscaleModel
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class UpscaleModelFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class VAE
{
    public object type { get; set; }
    public string name { get; set; }
    public object model { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class VAEFile
{
    public object type { get; set; }
    public string name { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class VideoRef
{
    public object type { get; set; }
    public string uri { get; set; }
    public object asset_id { get; set; }
    public object data { get; set; }
    public double? duration { get; set; }
    public string? format { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class WorkflowRef
{
    public object type { get; set; }
    public string id { get; set; }
}

[MessagePack.MessagePackObject(keyAsPropertyName: true)]
public partial class unCLIPFile
{
    public object type { get; set; }
    public string name { get; set; }
}

}
