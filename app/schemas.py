"""Pydantic request/response models for the anomaly-detection API."""
from pydantic import BaseModel, Field


class Connection(BaseModel):
    """A single network-connection record (UNSW-NB15 raw feature schema).

    All fields are optional with neutral defaults so partial records still
    score; supplying the full record gives the most accurate result.
    """
    dur: float = 0.0
    proto: str = "tcp"
    service: str = "-"
    state: str = "INT"
    spkts: int = 0
    dpkts: int = 0
    sbytes: int = 0
    dbytes: int = 0
    rate: float = 0.0
    sttl: int = 0
    dttl: int = 0
    sload: float = 0.0
    dload: float = 0.0
    sloss: int = 0
    dloss: int = 0
    sinpkt: float = 0.0
    dinpkt: float = 0.0
    sjit: float = 0.0
    djit: float = 0.0
    swin: int = 0
    stcpb: int = 0
    dtcpb: int = 0
    dwin: int = 0
    tcprtt: float = 0.0
    synack: float = 0.0
    ackdat: float = 0.0
    smean: int = 0
    dmean: int = 0
    trans_depth: int = 0
    response_body_len: int = 0
    ct_srv_src: int = 0
    ct_state_ttl: int = 0
    ct_dst_ltm: int = 0
    ct_src_dport_ltm: int = 0
    ct_dst_sport_ltm: int = 0
    ct_dst_src_ltm: int = 0
    is_ftp_login: int = 0
    ct_ftp_cmd: int = 0
    ct_flw_http_mthd: int = 0
    ct_src_ltm: int = 0
    ct_srv_dst: int = 0
    is_sm_ips_ports: int = 0

    model_config = {"extra": "ignore"}


class PredictionResponse(BaseModel):
    attack_probability: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., description="0 = normal, 1 = attack")
    label: str
    threshold: float
