import torch
import torch.nn.functional as F
from models.schedule import cosine_beta_schedule
from models.unet import ImprovedDiffusionUNet1D

class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        window_size: int,
        in_channels: int,
        time_emb_dim: int,
        base_channels: int,
        n_res_blocks: int,
        timesteps: int,
        s: float = 0.007,
    ):
        '''
        Args:
            window_size: length of each 1D window
            in_channels: number of feature channels D
            time_emb_dim: embedding dimension for time/cond
            base_channels: number of channels in U-Net
            n_res_blocks: number of residual blocks in U-Net
            timesteps: total diffusion steps (T)
            s: small offset in cosine schedule
        '''
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.T = timesteps

        # noise schedule
        betas, alphas, alpha_bars = cosine_beta_schedule(timesteps, s)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        # diffusion model (1D U-Net)
        self.model = ImprovedDiffusionUNet1D(
            in_channels=in_channels,
            window_size=window_size,
            time_emb_dim=time_emb_dim,
            base_channels=base_channels,
            n_res_blocks=n_res_blocks,
        )

    def forward(
        self,
        x0: torch.Tensor,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
    ) -> torch.Tensor:
        '''
        Compute training loss: sample a random timestep t for each example,
        add noise, predict the noise, and return the MSE loss.

        Args:
            x0: clean input tensor of shape (B, W, D)
            start_idx: window start indices, shape (B,)
            series_len: original series lengths, shape (B,)
        Returns:
            MSE loss between true and predicted noise
        '''
        # validate input shape
        assert x0.dim() == 3 and x0.size(2) == self.in_channels, f'Expected input shape (B, W, {self.in_channels}), got {tuple(x0.shape)}'
        # keep feature dim
        x0_flat = x0  # (B, W, D)

        B = x0_flat.size(0)
        device = x0_flat.device

        # sample random timesteps
        t = torch.randint(0, self.T, (B,), device=device)
        # sample noise
        eps = torch.randn_like(x0_flat)

        # compute noisy input x_t
        a_bar = self.alpha_bars[t].view(B, 1, 1)
        x_t = torch.sqrt(a_bar) * x0_flat + torch.sqrt(1 - a_bar) * eps

        # predict noise
        eps_pred = self.model(x_t, t, start_idx, series_len)

        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(
        self,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
        device: torch.device = None,
    ) -> torch.Tensor:
        '''
        Generate samples by reversing the diffusion process.

        Args:
            start_idx: window start indices, shape (B,)
            series_len: original series lengths, shape (B,)
            device: torch device (defaults to model's device)
        Returns:
            x0 samples: tensor of shape (B, W, D)
        '''
        if device is None:
            device = self.betas.device

        start_idx = start_idx.to(device)
        series_len = series_len.to(device)
        B = start_idx.size(0)

        # initialize with Gaussian noise
        x = torch.randn(B, self.window_size, self.in_channels, device=device)

        # reverse diffusion
        for t in reversed(range(self.T)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = self.model(x, t_tensor, start_idx, series_len)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            a_bar_t = self.alpha_bars[t]

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - a_bar_t)
            x_prev = coef1 * (x - coef2 * eps_pred)

            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = x_prev + sigma_t * noise
            else:
                x = x_prev

        return x

    @torch.no_grad()
    def sample_total(
        self,
        start_idx: torch.LongTensor,   # (B,)
        end_idx: torch.LongTensor,     # (B,)
        shift: int,
        min_value: float,
        max_value: float,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Sample a full series from start to end by sliding and averaging overlapping windows,
        ignoring out-of-range values.

        Args:
            start_idx: (B,) tensor of start positions
            end_idx:   (B,) tensor of end positions
            shift:     int stride (1 <= shift < window_size)
            min_value: float minimum valid value
            max_value: float maximum valid value
            device:    torch device (defaults to model's device)

        Returns:
            Tensor of shape (B, L, D) where L = end_idx - start_idx + 1
        """
        if device is None:
            device = self.betas.device

        # Move to device
        start_idx = start_idx.to(device)    # (B,)
        end_idx   = end_idx.to(device)      # (B,)
        B = start_idx.size(0)

        # Ensure all series lengths L are equal across batch
        lengths = end_idx - start_idx + 1   # (B,)
        if not torch.all(lengths == lengths[0]):
            raise ValueError("All series lengths must be equal across batch")
        L = int(lengths[0].item())          # scalar

        # Validate shift
        if not (1 <= shift < self.window_size):
            raise ValueError(f"shift must be between 1 and {self.window_size-1}")

        # Compute window start positions
        s0 = int(start_idx[0].item())
        e0 = s0 + L - 1
        positions = list(range(s0, e0 - self.window_size + 2, shift))

        # Prepare accumulators
        sum_series = torch.zeros(B, L, self.in_channels, device=device)
        # sum_series: (B, L, D)
        count      = torch.zeros(B, L, self.in_channels, device=device)
        # count:      (B, L, D)

        # Slide & sample each window
        for ws in positions:
            ws_tensor       = torch.full((B,), ws, dtype=torch.long, device=device)
            series_len_tens = torch.full((B,), L,  dtype=torch.long, device=device)

            # x_win: (B, window_size, D)
            x_win = self.sample(ws_tensor, series_len_tens, device=device)
            # mask out-of-range entries: (B, window_size, D)
            mask = (x_win >= min_value) & (x_win <= max_value)
            mask_f = mask.float()

            offset = ws - s0  # scalar offset into [0, L-window_size]
            # accumulate only valid entries
            sum_series[:, offset:offset+self.window_size, :] += x_win * mask_f
            count[:,      offset:offset+self.window_size, :] += mask_f

        # Compute final averaged series, safe divide
        avg = torch.where(count > 0, sum_series / count, torch.zeros_like(sum_series))
        # avg: (B, L, D)
        return avg
