using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MessagePack;
using Nodetool.Metadata;

namespace Nodetool
{
    public enum RunnerState
    {
        Idle,
        Connecting,
        Connected,
        Running,
        Error,
        Cancelled
    }

    public class WebSocketRunnerClient : IDisposable
    {
        private ClientWebSocket _socket = new ClientWebSocket();
        public RunnerState State { get; private set; } = RunnerState.Idle;
        public string? JobId { get; private set; }
        public string? StatusMessage { get; private set; }
        public Uri? CurrentUrl { get; private set; }

        public async Task ConnectAsync(string url, CancellationToken ct = default)
        {
            if (_socket.State == WebSocketState.Open)
            {
                await DisconnectAsync();
            }
            State = RunnerState.Connecting;
            CurrentUrl = new Uri(url);
            _socket = new ClientWebSocket();
            await _socket.ConnectAsync(CurrentUrl, ct);
            State = RunnerState.Connected;
        }

        public async Task DisconnectAsync()
        {
            if (_socket.State == WebSocketState.Open)
            {
                await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None);
            }
            State = RunnerState.Idle;
        }

        public async Task RunAsync(string workflowId, Dictionary<string, object> parameters, CancellationToken ct = default)
        {
            if (State != RunnerState.Connected)
            {
                throw new InvalidOperationException("WebSocket not connected");
            }

            var request = new Dictionary<string, object>
            {
                {"type", "run_job_request"},
                {"workflow_id", workflowId},
                {"job_type", "workflow"},
                {"auth_token", "local_token"},
                {"params", parameters}
            };

            var payload = MessagePackSerializer.Serialize(new Dictionary<string, object>
            {
                {"command", "run_job"},
                {"data", request}
            });

            await _socket.SendAsync(payload, WebSocketMessageType.Binary, true, ct);
            State = RunnerState.Running;
            await ReceiveLoop(ct);
        }

        private async Task ReceiveLoop(CancellationToken ct)
        {
            var buffer = new byte[1024 * 32];
            while (_socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var result = await _socket.ReceiveAsync(buffer, ct);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    break;
                }
                var data = MessagePackSerializer.Deserialize<Dictionary<string, object>>(buffer.AsMemory(0, result.Count));
                if (data.TryGetValue("type", out var t) && t is string type)
                {
                    if (type == "job_update")
                    {
                        if (data.TryGetValue("status", out var statusObj) && statusObj is string status)
                        {
                            StatusMessage = $"Job {status}";
                            if (status == "completed")
                            {
                                await DisconnectAsync();
                                break;
                            }
                            if (status == "failed")
                            {
                                State = RunnerState.Error;
                            }
                        }
                        if (data.TryGetValue("job_id", out var jid) && jid is string id)
                        {
                            JobId = id;
                        }
                    }
                }
            }
        }

        public void Dispose()
        {
            _socket.Dispose();
        }
    }
}
